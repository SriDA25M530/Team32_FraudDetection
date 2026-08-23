import os

import mlflow
import mlflow.spark

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    GBTClassifier,
)
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)


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

PROCESSED_DATA_DIR = os.path.join(
    BASE_PATH,
    "data",
    "processed"
)

MODEL_ROOT = os.path.join(
    BASE_PATH,
    "data",
    "models",
    DAG_ID
)

LR_MODEL_DIR = os.path.join(
    MODEL_ROOT,
    "logistic_regression"
)

GBT_MODEL_DIR = os.path.join(
    MODEL_ROOT,
    "gbt_classifier"
)


# =========================================================
# TRAINING
# =========================================================

def run_spark_training():

    print("==================================================")
    print("TASK - MODEL TRAINING STARTED")
    print("==================================================")

    print(f"DAG ID          : {DAG_ID}")
    print(f"Base Path       : {BASE_PATH}")
    print(f"Processed Data  : {PROCESSED_DATA_DIR}")
    print(f"Model Root      : {MODEL_ROOT}")
    print(f"LR Model        : {LR_MODEL_DIR}")
    print(f"GBT Model       : {GBT_MODEL_DIR}")
    print("")

    os.makedirs(
        MODEL_ROOT,
        exist_ok=True
    )

    # -----------------------------------------------------
    # Spark
    # -----------------------------------------------------

    spark = (
        SparkSession.builder
        .appName(
            f"{DAG_ID}_training"
        )
        .getOrCreate()
    )

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------

    print("[STEP 1] Loading processed dataset...")

    df = spark.read.csv(
        PROCESSED_DATA_DIR,
        header=True,
        inferSchema=True
    )

    total_count = df.count()

    print(
        f"[OK] Training dataset records: "
        f"{total_count:,}"
    )

    # -----------------------------------------------------
    # Feature assembly
    # -----------------------------------------------------

    print("[STEP 2] Assembling features...")

    feature_cols = [
        c for c in df.columns
        if c != "is_fraud"
    ]

    print(
        f"[INFO] Number of features: "
        f"{len(feature_cols)}"
    )

    print(
        f"[INFO] Features: {feature_cols}"
    )

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="skip"
    )

    df_assembled = assembler.transform(df)

    # -----------------------------------------------------
    # Split
    # -----------------------------------------------------

    print("[STEP 3] Splitting dataset 80/20...")

    train_df, test_df = (
        df_assembled.randomSplit(
            [0.8, 0.2],
            seed=42
        )
    )

    train_count = train_df.count()
    test_count = test_df.count()

    print(
        f"Training records: {train_count:,}"
    )

    print(
        f"Testing records : {test_count:,}"
    )

    # -----------------------------------------------------
    # Evaluators
    # -----------------------------------------------------

    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="is_fraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR",
    )

    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="f1",
    )

    # =====================================================
    # MODEL 1 - LOGISTIC REGRESSION
    # =====================================================

    print("")
    print("==================================================")
    print("MODEL 1 - LOGISTIC REGRESSION")
    print("==================================================")

    with mlflow.start_run(
        run_name=f"{DAG_ID}_LogisticRegression"
    ):

        lr = LogisticRegression(
            featuresCol="features",
            labelCol="is_fraud",
            maxIter=100,
        )

        print("[STEP] Training Logistic Regression...")

        lr_model = lr.fit(train_df)

        print("[OK] Logistic Regression training complete.")

        predictions = lr_model.transform(
            test_df
        )

        pr_auc = pr_evaluator.evaluate(
            predictions
        )

        f1 = f1_evaluator.evaluate(
            predictions
        )

        print(
            f"PR-AUC : {pr_auc:.4f}"
        )

        print(
            f"F1     : {f1:.4f}"
        )

        mlflow.log_param(
            "model_type",
            "LogisticRegression"
        )

        mlflow.log_param(
            "dag_id",
            DAG_ID
        )

        mlflow.log_metrics(
            {
                "pr_auc": pr_auc,
                "f1": f1,
            }
        )

        print(
            f"[STEP] Saving model to:"
        )

        print(LR_MODEL_DIR)

        lr_model.write().overwrite().save(
            LR_MODEL_DIR
        )

        print(
            "[OK] Logistic Regression model saved."
        )

    # =====================================================
    # MODEL 2 - GBT
    # =====================================================

    print("")
    print("==================================================")
    print("MODEL 2 - GBT CLASSIFIER")
    print("==================================================")

    with mlflow.start_run(
        run_name=f"{DAG_ID}_GBT"
    ):

        gbt = GBTClassifier(
            featuresCol="features",
            labelCol="is_fraud",
            maxIter=20,
            maxDepth=5,
        )

        print("[STEP] Training GBT Classifier...")

        gbt_model = gbt.fit(train_df)

        print("[OK] GBT training complete.")

        predictions = gbt_model.transform(
            test_df
        )

        pr_auc = pr_evaluator.evaluate(
            predictions
        )

        f1 = f1_evaluator.evaluate(
            predictions
        )

        print(
            f"PR-AUC : {pr_auc:.4f}"
        )

        print(
            f"F1     : {f1:.4f}"
        )

        mlflow.log_param(
            "model_type",
            "GBTClassifier"
        )

        mlflow.log_param(
            "dag_id",
            DAG_ID
        )

        mlflow.log_metrics(
            {
                "pr_auc": pr_auc,
                "f1": f1,
            }
        )

        print(
            "[STEP] Saving model to:"
        )

        print(GBT_MODEL_DIR)

        gbt_model.write().overwrite().save(
            GBT_MODEL_DIR
        )

        print(
            "[OK] GBT model saved."
        )

    # =====================================================
    # FINAL
    # =====================================================

    print("")
    print("==================================================")
    print("MODEL TRAINING COMPLETED")
    print("==================================================")

    print(f"Model root: {MODEL_ROOT}")

    print("")
    print("Models created:")

    print(
        f"1. Logistic Regression:"
    )
    print(
        f"   {LR_MODEL_DIR}"
    )

    print(
        f"2. GBT Classifier:"
    )
    print(
        f"   {GBT_MODEL_DIR}"
    )

    print("==================================================")

    spark.stop()


if __name__ == "__main__":
    run_spark_training()