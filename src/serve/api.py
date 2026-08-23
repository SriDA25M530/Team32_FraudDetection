import os
from typing import Dict

import uvicorn

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter
from pydantic import BaseModel

from pyspark.sql import SparkSession, Row

from pyspark.ml.classification import (
    GBTClassificationModel,
    LogisticRegressionModel,
)

from pyspark.ml.feature import VectorAssembler


# =========================================================
# CONFIGURATION
# =========================================================

BASE_PATH = os.getenv(
    "BASE_PATH",
    "/storage/uploads/test/Team32_FraudDetection"
)

DAG_ID = os.getenv(
    "DAG_ID",
    "team32_V2"
)

# ---------------------------------------------------------
# MODEL PATH
#
# Airflow/Docker can provide:
#
# MODEL_PATH=/storage/uploads/test/Team32_FraudDetection/data/models
#
# If MODEL_PATH is not provided, fall back to:
#
# BASE_PATH/data/models/DAG_ID
# ---------------------------------------------------------

MODEL_PATH_ENV = os.getenv("MODEL_PATH")

if MODEL_PATH_ENV:
    MODEL_ROOT = os.path.join(
        MODEL_PATH_ENV,
        DAG_ID
    )
else:
    MODEL_ROOT = os.path.join(
        BASE_PATH,
        "data",
        "models",
        DAG_ID
    )


# =========================================================
# MODEL DIRECTORIES
# =========================================================

LR_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "logistic_regression"
)

GBT_MODEL_PATH = os.path.join(
    MODEL_ROOT,
    "gbt_classifier"
)


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title=(
        "Team32 Fraud Detection API "
        f"- {DAG_ID}"
    )
)


# =========================================================
# PROMETHEUS
# =========================================================

Instrumentator().instrument(app).expose(app)


prediction_counter = Counter(
    "fraud_predictions_total",
    "Total number of fraud predictions",
    [
        "prediction_class",
        "model_name",
    ],
)


# =========================================================
# GLOBAL VARIABLES
# =========================================================

spark = None

lr_model = None

gbt_model = None


# =========================================================
# REQUEST MODEL
# =========================================================

class TransactionRequest(BaseModel):
    features: Dict[str, float]


# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get("/health")
def health():

    lr_loaded = lr_model is not None

    gbt_loaded = gbt_model is not None

    return {
        "status": (
            "healthy"
            if lr_loaded and gbt_loaded
            else "unhealthy"
        ),
        "dag_id": DAG_ID,
        "base_path": BASE_PATH,
        "model_root": MODEL_ROOT,
        "logistic_regression_loaded": lr_loaded,
        "gbt_classifier_loaded": gbt_loaded,
    }


# =========================================================
# STARTUP - LOAD MODELS
# =========================================================

@app.on_event("startup")
def load_models():

    global spark
    global lr_model
    global gbt_model

    print("==================================================")
    print("FASTAPI STARTUP")
    print("==================================================")

    print(f"DAG ID     : {DAG_ID}")
    print(f"Base Path  : {BASE_PATH}")
    print(f"Model Root : {MODEL_ROOT}")
    print("")

    print(f"LR Model  : {LR_MODEL_PATH}")

    print(f"GBT Model : {GBT_MODEL_PATH}")

    print("")

    # -----------------------------------------------------
    # CHECK MODEL ROOT
    # -----------------------------------------------------

    if not os.path.isdir(MODEL_ROOT):

        raise RuntimeError(
            f"Model root does not exist: "
            f"{MODEL_ROOT}"
        )

    # -----------------------------------------------------
    # CHECK LOGISTIC REGRESSION MODEL
    # -----------------------------------------------------

    if not os.path.isdir(LR_MODEL_PATH):

        raise RuntimeError(
            f"LR model does not exist: "
            f"{LR_MODEL_PATH}"
        )

    # -----------------------------------------------------
    # CHECK GBT MODEL
    # -----------------------------------------------------

    if not os.path.isdir(GBT_MODEL_PATH):

        raise RuntimeError(
            f"GBT model does not exist: "
            f"{GBT_MODEL_PATH}"
        )

    print("[OK] Model directories exist.")

    print("")
    print("[STEP] Starting Spark inference session...")

    # -----------------------------------------------------
    # START SPARK
    # -----------------------------------------------------

    spark = (
        SparkSession.builder
        .appName(
            f"{DAG_ID}_FraudInferenceAPI"
        )
        .master("local[1]")
        .getOrCreate()
    )

    print(
        "[OK] Spark inference session started."
    )

    # -----------------------------------------------------
    # LOAD LOGISTIC REGRESSION MODEL
    # -----------------------------------------------------

    try:

        print(
            "[STEP] Loading Logistic Regression model..."
        )

        lr_model = (
            LogisticRegressionModel
            .load(LR_MODEL_PATH)
        )

        print(
            "[OK] Logistic Regression loaded."
        )

    except Exception as e:

        print(
            "[ERROR] Logistic Regression loading failed:"
        )

        print(str(e))

        raise

    # -----------------------------------------------------
    # LOAD GBT MODEL
    # -----------------------------------------------------

    try:

        print(
            "[STEP] Loading GBT model..."
        )

        gbt_model = (
            GBTClassificationModel
            .load(GBT_MODEL_PATH)
        )

        print(
            "[OK] GBT model loaded."
        )

    except Exception as e:

        print(
            "[ERROR] GBT model loading failed:"
        )

        print(str(e))

        raise

    # -----------------------------------------------------
    # STARTUP COMPLETE
    # -----------------------------------------------------

    print("")

    print("==================================================")
    print("FASTAPI STARTUP COMPLETED")
    print("==================================================")


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict_fraud(
    req: TransactionRequest
):

    # -----------------------------------------------------
    # CHECK MODELS
    # -----------------------------------------------------

    if lr_model is None or gbt_model is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Models failed to load "
                "during startup."
            ),
        )

    # -----------------------------------------------------
    # CHECK SPARK
    # -----------------------------------------------------

    if spark is None:

        raise HTTPException(
            status_code=500,
            detail="Spark session is not available.",
        )

    try:

        # -------------------------------------------------
        # CHECK FEATURES
        # -------------------------------------------------

        if not req.features:

            raise ValueError(
                "No features were supplied."
            )

        # -------------------------------------------------
        # CREATE SPARK ROW
        # -------------------------------------------------

        row = Row(
            **req.features
        )

        # -------------------------------------------------
        # CREATE SPARK DATAFRAME
        # -------------------------------------------------

        df = spark.createDataFrame(
            [row]
        )

        # -------------------------------------------------
        # FEATURE COLUMNS
        # -------------------------------------------------

        feature_cols = list(
            req.features.keys()
        )

        # -------------------------------------------------
        # VECTOR ASSEMBLER
        # -------------------------------------------------

        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="features",
            handleInvalid="skip",
        )

        df_assembled = (
            assembler.transform(df)
        )

        # -------------------------------------------------
        # LOGISTIC REGRESSION PREDICTION
        # -------------------------------------------------

        lr_predictions = (
            lr_model.transform(
                df_assembled
            )
        )

        lr_result = (
            lr_predictions
            .select(
                "probability",
                "prediction"
            )
            .first()
        )

        # -------------------------------------------------
        # GBT PREDICTION
        # -------------------------------------------------

        gbt_predictions = (
            gbt_model.transform(
                df_assembled
            )
        )

        gbt_result = (
            gbt_predictions
            .select(
                "probability",
                "prediction"
            )
            .first()
        )

        # -------------------------------------------------
        # EXTRACT LR PREDICTION
        # -------------------------------------------------

        lr_is_fraud = int(
            lr_result.prediction
        )

        lr_probability = float(
            lr_result.probability[1]
        )

        # -------------------------------------------------
        # EXTRACT GBT PREDICTION
        # -------------------------------------------------

        gbt_is_fraud = int(
            gbt_result.prediction
        )

        gbt_probability = float(
            gbt_result.probability[1]
        )

        # -------------------------------------------------
        # PROMETHEUS - LOGISTIC REGRESSION
        # -------------------------------------------------

        prediction_counter.labels(
            prediction_class=(
                "fraud"
                if lr_is_fraud == 1
                else "genuine"
            ),
            model_name="logistic_regression",
        ).inc()

        # -------------------------------------------------
        # PROMETHEUS - GBT
        # -------------------------------------------------

        prediction_counter.labels(
            prediction_class=(
                "fraud"
                if gbt_is_fraud == 1
                else "genuine"
            ),
            model_name="gbt",
        ).inc()

        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {
           
            "logistic_regression": {
                "fraud_probability":
                    lr_probability,

                "is_fraud_prediction":
                    lr_is_fraud,
            },

            "gbt_classifier": {
                "fraud_probability":
                    gbt_probability,

                "is_fraud_prediction":
                    gbt_is_fraud,
            },
        }

    except HTTPException:

        raise

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "application": "Team32 Fraud Detection API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
        },
    }


# =========================================================
# LOCAL EXECUTION
# =========================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )