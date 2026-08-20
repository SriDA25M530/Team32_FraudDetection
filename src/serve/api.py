from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
import uvicorn
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel, LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import Row

app = FastAPI(title="Team 32 - Fraud Detection API (Model Comparison)")

# Paths to both saved models
LR_MODEL_PATH = "/storage/scratch/Team32_FraudDetection/logistic_regression_model"
GBT_MODEL_PATH = "/storage/scratch/Team32_FraudDetection/gbt_classifier_model"

spark = None
lr_model = None
gbt_model = None

class TransactionRequest(BaseModel):
    # Expects a dictionary of the preprocessed features 
    features: Dict[str, float]

@app.on_event("startup")
def load_models():
    global spark, lr_model, gbt_model
    # Initialize a lightweight local Spark session for inference
    spark = SparkSession.builder \
        .appName("FraudInferenceAPI") \
        .master("local[1]") \
        .getOrCreate()
    
    try:
        lr_model = LogisticRegressionModel.load(LR_MODEL_PATH)
        print(f"Logistic Regression model loaded from {LR_MODEL_PATH}")
        
        gbt_model = GBTClassificationModel.load(GBT_MODEL_PATH)
        print(f"GBT Classifier model loaded from {GBT_MODEL_PATH}")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.post("/predict")
def predict_fraud(req: TransactionRequest):
    if not lr_model or not gbt_model:
        raise HTTPException(status_code=500, detail="Models failed to load on startup.")

    try:
        # 1. Convert the JSON dictionary to a PySpark DataFrame
        row = Row(**req.features)
        df = spark.createDataFrame([row])
        
        # 2. Assemble the features into a single Vector column
        feature_cols = list(req.features.keys())
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        df_assembled = assembler.transform(df)
        
        # 3. Run Inference on both models
        lr_predictions = lr_model.transform(df_assembled)
        gbt_predictions = gbt_model.transform(df_assembled)
        
        # 4. Extract probabilities and class predictions
        lr_result = lr_predictions.select("probability", "prediction").first()
        gbt_result = gbt_predictions.select("probability", "prediction").first()
        
        return {
            "logistic_regression": {
                "fraud_probability": float(lr_result.probability[1]),
                "is_fraud_prediction": int(lr_result.prediction)
            },
            "gbt_classifier": {
                "fraud_probability": float(gbt_result.probability[1]),
                "is_fraud_prediction": int(gbt_result.prediction)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)