import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import mlflow
import mlflow.spark

# Server Paths
PROCESSED_DATA_DIR = "/storage/uploads/Team32_FraudDetection/data/processed/"
LR_MODEL_DIR = "/storage/scratch/Team32_FraudDetection/logistic_regression_model"
GBT_MODEL_DIR = "/storage/scratch/Team32_FraudDetection/gbt_classifier_model"

def run_spark_training():
    print("==================================================")
    print("   Starting PySpark Model Training                ")
    print("==================================================")

    spark = SparkSession.builder \
        .appName("team32_model_training") \
        .getOrCreate()
    
    print(f"Loading data from {PROCESSED_DATA_DIR}...")
    df = spark.read.csv(PROCESSED_DATA_DIR, header=True, inferSchema=True)
    
    print("Assembling feature vector...")
    feature_cols = [c for c in df.columns if c != 'is_fraud']
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    df_assembled = assembler.transform(df)
    
    print("Splitting data (80/20)...")
    train_df, test_df = df_assembled.randomSplit([0.8, 0.2], seed=42)
    
    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
    )
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol="is_fraud", predictionCol="prediction", metricName="f1"
    )

    # ---------------------------------------------------------
    # MODEL 1: Logistic Regression
    # ---------------------------------------------------------
    with mlflow.start_run(run_name="Spark_LR_Baseline"):
        print("\nTraining PySpark Logistic Regression...")
        lr = LogisticRegression(featuresCol="features", labelCol="is_fraud", maxIter=100)
        lr_model = lr.fit(train_df)
        
        predictions = lr_model.transform(test_df)
        pr_auc = pr_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)
        
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metrics({"pr_auc": pr_auc, "macro_f1": f1})
        
        print(f"LR Metrics -> PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}")
        
        print(f"Saving Logistic Regression model to {LR_MODEL_DIR}...")
        lr_model.write().overwrite().save(LR_MODEL_DIR)

    # ---------------------------------------------------------
    # MODEL 2: GBTClassifier
    # ---------------------------------------------------------
    with mlflow.start_run(run_name="Spark_GBT_Advanced"):
        print("\nTraining PySpark GBTClassifier...")
        gbt = GBTClassifier(featuresCol="features", labelCol="is_fraud", maxIter=20, maxDepth=5)
        gbt_model = gbt.fit(train_df)
        
        predictions = gbt_model.transform(test_df)
        pr_auc = pr_evaluator.evaluate(predictions)
        f1 = f1_evaluator.evaluate(predictions)
        
        mlflow.log_param("model_type", "GBTClassifier")
        mlflow.log_metrics({"pr_auc": pr_auc, "macro_f1": f1})
        
        print(f"GBT Metrics -> PR-AUC: {pr_auc:.4f} | F1: {f1:.4f}")
        
        print(f"Saving GBT Classifier model to {GBT_MODEL_DIR}...")
        gbt_model.write().overwrite().save(GBT_MODEL_DIR)
    
    print("\n[Success] PySpark Training & Local Saving Complete!")
    print("==================================================")
    
    spark.stop()

if __name__ == "__main__":
    run_spark_training()