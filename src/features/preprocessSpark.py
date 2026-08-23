import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import RobustScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array


# =========================================================
# CONFIGURATION
# =========================================================

BASE_PATH = os.getenv(
    "BASE_PATH",
    "/storage/uploads/Team32_FraudDetection"
)

DAG_ID = os.getenv(
    "DAG_ID",
    "team32_final_end_to_end_pipeline_V1"
)

INPUT_CSV = os.path.join(
    BASE_PATH,
    "data",
    "raw",
    f"{DAG_ID}_kafka_consumed.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_PATH,
    "data",
    "processed"
)


# =========================================================
# PREPROCESSING
# =========================================================

def run_pyspark_preprocessing():

    print("==================================================")
    print("TASK - PYSPARK PREPROCESSING STARTED")
    print("==================================================")

    print(f"Base Path   : {BASE_PATH}")
    print(f"DAG ID      : {DAG_ID}")
    print(f"Input       : {INPUT_CSV}")
    print(f"Output      : {OUTPUT_DIR}")
    print("")

    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Spark
    # -----------------------------------------------------

    spark = (
        SparkSession.builder
        .appName(f"{DAG_ID}_preprocessing")
        .getOrCreate()
    )

    # -----------------------------------------------------
    # Read
    # -----------------------------------------------------

    print("[STEP 1] Loading input data...")

    df = spark.read.csv(
        INPUT_CSV,
        header=True,
        inferSchema=True
    )

    input_count = df.count()

    print(
        f"[OK] Input records: {input_count:,}"
    )

    # -----------------------------------------------------
    # Time features
    # -----------------------------------------------------

    print("[STEP 2] Creating time features...")

    df = df.withColumn(
        "trans_date_trans_time",
        F.to_timestamp(
            "trans_date_trans_time"
        )
    )

    df = df.withColumn(
        "trans_hour",
        F.hour("trans_date_trans_time")
    )

    df = df.withColumn(
        "trans_day_of_week",
        F.dayofweek(
            "trans_date_trans_time"
        )
    )

    # -----------------------------------------------------
    # Haversine
    # -----------------------------------------------------

    print("[STEP 3] Calculating geographic distance...")

    df = (
        df
        .withColumn(
            "lat1",
            F.radians(F.col("lat"))
        )
        .withColumn(
            "lon1",
            F.radians(F.col("long"))
        )
        .withColumn(
            "lat2",
            F.radians(F.col("merch_lat"))
        )
        .withColumn(
            "lon2",
            F.radians(F.col("merch_long"))
        )
    )

    df = (
        df
        .withColumn(
            "dlat",
            F.col("lat2") - F.col("lat1")
        )
        .withColumn(
            "dlon",
            F.col("lon2") - F.col("lon1")
        )
    )

    a = (
        F.pow(
            F.sin(F.col("dlat") / 2),
            2
        )
        +
        F.cos(F.col("lat1"))
        * F.cos(F.col("lat2"))
        * F.pow(
            F.sin(F.col("dlon") / 2),
            2
        )
    )

    df = df.withColumn(
        "distance_km",
        6371.0
        * 2
        * F.asin(F.sqrt(a))
    )

    # -----------------------------------------------------
    # Categorical encoding
    # -----------------------------------------------------

    print("[STEP 4] Encoding categorical features...")

    if "gender" in df.columns:

        df = df.withColumn(
            "gender_M",
            F.when(
                F.col("gender") == "M",
                1
            ).otherwise(0)
        )

    if "category" in df.columns:

        categories = [
            row["category"]
            for row in
            df.select("category")
            .distinct()
            .collect()
        ]

        for cat in categories[1:]:

            clean_cat_name = (
                f"cat_"
                f"{str(cat).replace(' ', '_').replace('/', '_')}"
            )

            df = df.withColumn(
                clean_cat_name,
                F.when(
                    F.col("category") == cat,
                    1
                ).otherwise(0)
            )

    # -----------------------------------------------------
    # Drop unnecessary columns
    # -----------------------------------------------------

    print("[STEP 5] Removing unnecessary columns...")

    cols_to_drop = [
        "_c0",
        "Unnamed: 0",
        "trans_date_trans_time",
        "merchant",
        "first",
        "last",
        "street",
        "city",
        "state",
        "zip",
        "job",
        "dob",
        "trans_num",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "category",
        "gender",
        "lat1",
        "lon1",
        "lat2",
        "lon2",
        "dlat",
        "dlon",
    ]

    df = df.drop(
        *[
            c for c in cols_to_drop
            if c in df.columns
        ]
    )

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------

    print("[STEP 6] Scaling numerical features...")

    assembler = VectorAssembler(
        inputCols=[
            "amt",
            "distance_km"
        ],
        outputCol="features_to_scale"
    )

    df = assembler.transform(df)

    scaler = RobustScaler(
        inputCol="features_to_scale",
        outputCol="scaled_features"
    )

    scaler_model = scaler.fit(df)

    df = scaler_model.transform(df)

    df = df.withColumn(
        "scaled_arr",
        vector_to_array("scaled_features")
    )

    df = df.withColumn(
        "amt_scaled",
        F.col("scaled_arr")[0]
    )

    df = df.withColumn(
        "distance_km_scaled",
        F.col("scaled_arr")[1]
    )

    df = df.drop(
        "amt",
        "distance_km",
        "features_to_scale",
        "scaled_features",
        "scaled_arr"
    )

    # -----------------------------------------------------
    # Class imbalance
    # -----------------------------------------------------

    print(
        "[STEP 7] Handling class imbalance..."
    )

    fraud_df = df.filter(
        F.col("is_fraud") == 1
    )

    legit_df = df.filter(
        F.col("is_fraud") == 0
    )

    fraud_count = fraud_df.count()
    legit_count = legit_df.count()

    print(
        f"Original Legit : {legit_count:,}"
    )

    print(
        f"Original Fraud : {fraud_count:,}"
    )

    if fraud_count == 0:
        raise RuntimeError(
            "No fraud records found."
        )

    oversample_ratio = (
        legit_count / float(fraud_count)
    )

    print(
        f"Oversampling ratio: "
        f"{oversample_ratio:.4f}"
    )

    oversampled_fraud_df = (
        fraud_df
        .sample(
            withReplacement=True,
            fraction=oversample_ratio,
            seed=42
        )
    )

    balanced_df = (
        legit_df
        .unionAll(
            oversampled_fraud_df
        )
    )

    balanced_count = balanced_df.count()

    balanced_fraud_count = (
        oversampled_fraud_df.count()
    )

    print(
        f"Balanced Legit : {legit_count:,}"
    )

    print(
        f"Balanced Fraud : {balanced_fraud_count:,}"
    )

    print(
        f"Balanced Total : {balanced_count:,}"
    )

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    print("")
    print("[STEP 8] Saving processed dataset...")

    balanced_df.coalesce(1).write.csv(
        OUTPUT_DIR,
        header=True,
        mode="overwrite"
    )

    print("")
    print("==================================================")
    print("PYSPARK PREPROCESSING COMPLETED")
    print("==================================================")

    print(
        f"Input records    : {input_count:,}"
    )

    print(
        f"Output records   : {balanced_count:,}"
    )

    print(
        f"Output directory : {OUTPUT_DIR}"
    )

    print("==================================================")

    spark.stop()


if __name__ == "__main__":
    run_pyspark_preprocessing()