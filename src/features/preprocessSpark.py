import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import RobustScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array

# Server Paths
INPUT_CSV = "/storage/uploads/Team32_FraudDetection/data/raw/team32_consumed_19th_Aug_3.csv"

# Note: Spark writes directories, not single files. We will output to a folder.
OUTPUT_DIR = "/storage/scratch/team32_processed_transactions_spark"

def run_pyspark_preprocessing():
    print("==================================================")
    print("   Starting PySpark Preprocessing Job             ")
    print("==================================================")
    
    # 1. Initialize Spark Session
    spark = SparkSession.builder \
        .appName("fraud_preprocessing") \
        .getOrCreate()
        
    print(f"Loading data from {INPUT_CSV}...")
    df = spark.read.csv(INPUT_CSV, header=True, inferSchema=True)
    
    # 2. Time-based Features
    print("Extracting time-based features...")
    df = df.withColumn("trans_date_trans_time", F.to_timestamp("trans_date_trans_time"))
    df = df.withColumn("trans_hour", F.hour("trans_date_trans_time"))
    # PySpark dayofweek returns 1 (Sunday) to 7 (Saturday)
    df = df.withColumn("trans_day_of_week", F.dayofweek("trans_date_trans_time"))

    # 3. Haversine Distance Calculation (Native PySpark Math)
    print("Calculating geographical distance (Haversine)...")
    df = df.withColumn("lat1", F.radians(F.col("lat"))) \
           .withColumn("lon1", F.radians(F.col("long"))) \
           .withColumn("lat2", F.radians(F.col("merch_lat"))) \
           .withColumn("lon2", F.radians(F.col("merch_long")))
           
    df = df.withColumn("dlat", F.col("lat2") - F.col("lat1"))
    df = df.withColumn("dlon", F.col("lon2") - F.col("lon1"))
    
    a = F.pow(F.sin(F.col("dlat") / 2), 2) + \
        F.cos(F.col("lat1")) * F.cos(F.col("lat2")) * F.pow(F.sin(F.col("dlon") / 2), 2)
        
    df = df.withColumn("distance_km", 6371.0 * 2 * F.asin(F.sqrt(a)))
    
    # 4. Encoding Categorical Features (Manual Dummies for easy CSV writing)
    print("Encoding categorical features...")
    if "gender" in df.columns:
        # Assuming M/F, drop one to avoid collinearity
        df = df.withColumn("gender_M", F.when(F.col("gender") == 'M', 1).otherwise(0))
        
    if "category" in df.columns:
        categories = [row['category'] for row in df.select('category').distinct().collect()]
        # Skip the first category to drop one column (Dummy Variable Trap)
        for cat in categories[1:]: 
            clean_cat_name = f"cat_{cat.replace(' ', '_').replace('/', '_')}"
            df = df.withColumn(clean_cat_name, F.when(F.col("category") == cat, 1).otherwise(0))

    # 5. Drop Unnecessary Columns
    print("Dropping unnecessary columns...")
    cols_to_drop = [
        '_c0', 'Unnamed: 0', 'trans_date_trans_time', 'merchant', 'first', 'last', 
        'street', 'city', 'state', 'zip', 'job', 'dob', 'trans_num', 
        'lat', 'long', 'merch_lat', 'merch_long', 'category', 'gender',
        'lat1', 'lon1', 'lat2', 'lon2', 'dlat', 'dlon' # dropping intermediate math columns
    ]
    df = df.drop(*[c for c in cols_to_drop if c in df.columns])

    # 6. Scaling Numerical Columns using RobustScaler
    print("Scaling numerical columns using PySpark RobustScaler...")
    assembler = VectorAssembler(inputCols=["amt", "distance_km"], outputCol="features_to_scale")
    df = assembler.transform(df)
    
    scaler = RobustScaler(inputCol="features_to_scale", outputCol="scaled_features")
    scaler_model = scaler.fit(df)
    df = scaler_model.transform(df)
    
    # Convert the vector back to individual columns to save as CSV
    df = df.withColumn("scaled_arr", vector_to_array("scaled_features"))
    df = df.withColumn("amt_scaled", F.col("scaled_arr")[0])
    df = df.withColumn("distance_km_scaled", F.col("scaled_arr")[1])
    
    # Drop the original unscaled columns and vector columns
    df = df.drop("amt", "distance_km", "features_to_scale", "scaled_features", "scaled_arr")

    # 7. Handle Class Imbalance via Random Oversampling
    print("Addressing class imbalance (Oversampling minority class)...")
    fraud_df = df.filter(F.col("is_fraud") == 1)
    legit_df = df.filter(F.col("is_fraud") == 0)
    
    fraud_count = fraud_df.count()
    legit_count = legit_df.count()
    
    print(f"Original counts -> Legit: {legit_count}, Fraud: {fraud_count}")
    
    # Calculate ratio to duplicate the fraud rows until they match legit rows
    oversample_ratio = legit_count / float(fraud_count)
    oversampled_fraud_df = fraud_df.sample(withReplacement=True, fraction=oversample_ratio, seed=42)
    
    balanced_df = legit_df.unionAll(oversampled_fraud_df)
    
    print(f"Balanced counts -> Legit: {legit_df.count()}, Fraud: {oversampled_fraud_df.count()}")

    # 8. Save Processed Data
    print(f"Saving processed dataset to {OUTPUT_DIR}...")
    # coalesce(1) forces Spark to write everything into a single CSV file inside the output directory
    balanced_df.coalesce(1).write.csv(OUTPUT_DIR, header=True, mode="overwrite")
    
    print("[Success] PySpark Preprocessing Complete!")
    print("==================================================")
    
    spark.stop()

if __name__ == "__main__":
    run_pyspark_preprocessing()