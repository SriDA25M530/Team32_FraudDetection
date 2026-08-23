from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


# ============================================================
# CONFIGURATION
# ============================================================

DAG_ID = "team32_final_end_to_end_pipeline_V1"

BASE_PATH = "/storage/uploads/Team32_FraudDetection"

RAW_DATA = f"{BASE_PATH}/data/raw/fraudTrain.csv"
RAW_TEST_DATA = f"{BASE_PATH}/data/raw/fraudTest.csv"

PRODUCER_SCRIPT = f"{BASE_PATH}/src/data_ingestion/kafka_producer.py"
CONSUMER_SCRIPT = f"{BASE_PATH}/src/data_ingestion/kafka_consumer.py"

PROCESSED_PATH = f"{BASE_PATH}/data/processed"
MODEL_PATH = f"{BASE_PATH}/data/models"

KAFKA_TOPIC = "team32_V007_topic"

API_CONTAINER = "team32_v007_fraud_api"
API_IMAGE = "team32_v007_fraud_api_image"


# ============================================================
# FASTAPI PORT CONFIGURATION
# ============================================================

API_PORT = 8009
API_INTERNAL_PORT = 8000

MLFLOW_URI = "http://mlflow-server:5000"


# ============================================================
# DEFAULT ARGUMENTS
# ============================================================

default_args = {
    "owner": "da25m512",
    "depends_on_past": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=2),
}


# ============================================================
# DAG
# ============================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description="Team32 end-to-end fraud detection pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["team32", "fraud-detection", "spark", "kafka", "mlflow"],
) as dag:

    # ========================================================
    # TASK 0 - INITIALIZATION
    # ========================================================

    initialize_pipeline = BashOperator(
        task_id="initialize_pipeline",
        bash_command=f"""
set -e

echo "============================================================"
echo "PIPELINE INITIALIZATION"
echo "============================================================"

echo "DAG ID              : {DAG_ID}"
echo "BASE PATH           : {BASE_PATH}"
echo "RAW DATA            : {RAW_DATA}"
echo "KAFKA TOPIC         : {KAFKA_TOPIC}"
echo "API CONTAINER       : {API_CONTAINER}"
echo "API IMAGE           : {API_IMAGE}"
echo "API HOST PORT       : {API_PORT}"
echo "API INTERNAL PORT   : {API_INTERNAL_PORT}"
echo "MODEL DIRECTORY     : {MODEL_PATH}"
echo "MLFLOW URI          : {MLFLOW_URI}"
echo ""

echo "[CHECK] Creating required directories..."

mkdir -p "{BASE_PATH}/data/raw"
mkdir -p "{BASE_PATH}/data/processed"
mkdir -p "{BASE_PATH}/data/models"
mkdir -p "{BASE_PATH}/logs"

echo "[OK] Required directories exist."
echo ""

echo "[CHECK] Checking Docker..."

if ! command -v docker >/dev/null 2>&1; then
    echo "[ERROR] Docker command is not available."
    exit 1
fi

echo "[OK] Docker command available."
echo ""

echo "[CHECK] Checking Spark master..."

SPARK_MASTER_ID=$(docker ps -q -f "name=^spark-master$")

if [ -z "$SPARK_MASTER_ID" ]; then
    echo "[ERROR] Spark master is not running."
    docker ps -a || true
    exit 1
fi

echo "[OK] Spark master is running."
echo "[INFO] Spark master container ID: $SPARK_MASTER_ID"
echo ""

echo "[CHECK] Checking BASE_PATH inside Spark master..."

if docker exec "$SPARK_MASTER_ID" test -d "{BASE_PATH}"; then
    echo "[OK] BASE_PATH is mounted inside Spark master."
else
    echo "[ERROR] BASE_PATH is NOT available inside Spark master."
    exit 1
fi

echo ""

echo "[CHECK] Checking raw CSV inside Spark master..."

if docker exec "$SPARK_MASTER_ID" test -f "{RAW_DATA}"; then
    echo "[OK] Raw CSV exists:"
    echo "{RAW_DATA}"
else
    echo "[ERROR] Raw CSV does not exist:"
    echo "{RAW_DATA}"
    exit 1
fi

echo ""

echo "[CHECK] Checking spark-submit..."

if docker exec "$SPARK_MASTER_ID" test -x /opt/spark/bin/spark-submit; then
    echo "[OK] spark-submit is available."
else
    echo "[ERROR] spark-submit was not found."
    exit 1
fi

echo ""

echo "============================================================"
echo "PIPELINE INITIALIZATION COMPLETED"
echo "============================================================"
""",
    )

    # ========================================================
    # TASK 1 - KAFKA PRODUCER
    # ========================================================

    team32_kp = BashOperator(
        task_id="team32_kp",
        bash_command=f"""
set -e

echo "============================================================"
echo "TASK 1 STARTED - KAFKA PRODUCER"
echo "============================================================"

SPARK_MASTER_ID=$(docker ps -q -f "name=^spark-master$")

if [ -z "$SPARK_MASTER_ID" ]; then
    echo "[ERROR] Spark master is not running."
    exit 1
fi

echo "[OK] Spark master: $SPARK_MASTER_ID"

if ! docker exec "$SPARK_MASTER_ID" test -f "{RAW_DATA}"; then
    echo "[ERROR] Input CSV not found:"
    echo "{RAW_DATA}"
    exit 1
fi

if ! docker exec "$SPARK_MASTER_ID" test -f "{PRODUCER_SCRIPT}"; then
    echo "[ERROR] Producer script not found:"
    echo "{PRODUCER_SCRIPT}"
    exit 1
fi

echo "[OK] Input CSV exists."
echo "[OK] Producer script exists."
echo ""

echo "[RUN] Starting Kafka producer..."

docker exec \
    -e DAG_ID="{DAG_ID}" \
    -e BASE_PATH="{BASE_PATH}" \
    -e KAFKA_TOPIC="{KAFKA_TOPIC}" \
    "$SPARK_MASTER_ID" \
    /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --total-executor-cores 2 \
    --executor-cores 2 \
    --executor-memory 4g \
    --driver-memory 1g \
    --conf spark.driver.cores=1 \
    --conf spark.driver.maxResultSize=1g \
    --conf spark.executor.memoryOverhead=384 \
    --conf spark.memory.fraction=0.600 \
    --conf spark.memory.storageFraction=0.500 \
    --conf spark.sql.shuffle.partitions=16 \
    --conf spark.default.parallelism=12 \
    --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128MB \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.jars.ivy=/tmp/.ivy2 \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 \
    --name "{DAG_ID}_producer" \
    "{PRODUCER_SCRIPT}"

echo ""
echo "[OK] Kafka producer completed."

echo ""
echo "============================================================"
echo "TASK 1 COMPLETED - KAFKA PRODUCER"
echo "============================================================"
""",
    )

    # ========================================================
    # TASK 2 - KAFKA CONSUMER
    # ========================================================

    team32_kc = BashOperator(
        task_id="team32_kc",
        bash_command=f"""
set -e

echo "============================================================"
echo "TASK 2 STARTED - KAFKA CONSUMER"
echo "============================================================"

SPARK_MASTER_ID=$(docker ps -q -f "name=^spark-master$")

if [ -z "$SPARK_MASTER_ID" ]; then
    echo "[ERROR] Spark master is not running."
    exit 1
fi

echo "[OK] Spark master: $SPARK_MASTER_ID"

if ! docker exec "$SPARK_MASTER_ID" test -f "{CONSUMER_SCRIPT}"; then
    echo "[ERROR] Consumer script not found:"
    echo "{CONSUMER_SCRIPT}"
    exit 1
fi

echo "[OK] Consumer script exists."
echo ""

echo "[RUN] Starting Kafka consumer..."

docker exec \
    -e DAG_ID="{DAG_ID}" \
    -e BASE_PATH="{BASE_PATH}" \
    -e KAFKA_TOPIC="{KAFKA_TOPIC}" \
    "$SPARK_MASTER_ID" \
    /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --total-executor-cores 2 \
    --executor-cores 2 \
    --executor-memory 4g \
    --driver-memory 1g \
    --conf spark.driver.cores=1 \
    --conf spark.driver.maxResultSize=1g \
    --conf spark.executor.memoryOverhead=384 \
    --conf spark.memory.fraction=0.600 \
    --conf spark.memory.storageFraction=0.500 \
    --conf spark.sql.shuffle.partitions=16 \
    --conf spark.default.parallelism=12 \
    --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128MB \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.jars.ivy=/tmp/.ivy2 \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 \
    --name "{DAG_ID}_consumer" \
    "{CONSUMER_SCRIPT}"

echo ""
echo "[OK] Kafka consumer completed."

echo ""
echo "============================================================"
echo "TASK 2 COMPLETED - KAFKA CONSUMER"
echo "============================================================"
""",
    )

    # ========================================================
    # TASK 3 - VALIDATE RAW DATA
    # ========================================================

    validate_raw_data = BashOperator(
        task_id="validate_raw_data",
        bash_command=f"""
set -e

echo "============================================================"
echo "TASK 3 - VALIDATE RAW DATA"
echo "============================================================"

SPARK_MASTER_ID=$(docker ps -q -f "name=^spark-master$")

if [ -z "$SPARK_MASTER_ID" ]; then
    echo "[ERROR] Spark master is not running."
    exit 1
fi

echo "[CHECK] Training data..."

docker exec "$SPARK_MASTER_ID" test -f "{RAW_DATA}"

echo "[OK] Training data exists."

if docker exec "$SPARK_MASTER_ID" test -f "{RAW_TEST_DATA}"; then
    echo "[OK] Test data exists."
else
    echo "[WARNING] Test data does not exist."
fi

echo ""
echo "[CHECK] Raw data directory..."

docker exec "$SPARK_MASTER_ID" sh -c \
    'ls -lh "{BASE_PATH}/data/raw/"'

echo ""
echo "============================================================"
echo "TASK 3 COMPLETED"
echo "============================================================"
""",
    )

    # ========================================================
    # TASK 4 - PREPROCESSING
    # ========================================================

    run_preprocessing = BashOperator(
        task_id="run_preprocessing",
        bash_command=f"""
set -e

echo "============================================================"
echo "TASK 4 - RUN PREPROCESSING"
echo "============================================================"

SPARK_MASTER_ID=$(docker ps -q -f "name=^spark-master$")

if [ -z "$SPARK_MASTER_ID" ]; then
    echo "[ERROR] Spark master is not running."
    exit 1
fi

PREPROCESS_SCRIPT="{BASE_PATH}/src/features/preprocessSpark.py"

if ! docker exec "$SPARK_MASTER_ID" test -f "$PREPROCESS_SCRIPT"; then
    echo "[ERROR] Preprocessing script not found:"
    echo "$PREPROCESS_SCRIPT"
    exit 1
fi

echo "[OK] Preprocessing script exists."

docker exec \
    -e DAG_ID="{DAG_ID}" \
    -e BASE_PATH="{BASE_PATH}" \
    "$SPARK_MASTER_ID" \
    /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --total-executor-cores 2 \
    --executor-cores 2 \
    --executor-memory 4g \
    --driver-memory 1g \
    --conf spark.driver.cores=1 \
    --conf spark.driver.maxResultSize=1g \
    --conf spark.executor.memoryOverhead=384 \
    --conf spark.sql.shuffle.partitions=16 \
    --conf spark.default.parallelism=12 \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.jars.ivy=/tmp/.ivy2 \
    --name "{DAG_ID}_preprocessing" \
    "$PREPROCESS_SCRIPT"

echo ""
echo "[OK] Preprocessing completed."

echo "============================================================"
echo "TASK 4 COMPLETED"
echo "============================================================"
""",
    )

    # ========================================================
    # TASK 5 - VALIDATE PROCESSED DATA
    # ========================================================

    validate_processed_data = BashOperator(
        task_id="validate_processed_data",
        bash_command=f"""
set -e

echo "============================================================"
echo "TASK 5 - VALIDATE PROCESSED DATA"
echo "============================================================"

SPARK_MASTER_ID=$(docker ps -q -f "name=^spark-master$")

if [ -z "$SPARK_MASTER_ID" ]; then
    echo "[ERROR] Spark master is not running."
    exit 1
fi

if ! docker exec "$SPARK_MASTER_ID" test -d "{PROCESSED_PATH}"; then
    echo "[ERROR] Processed directory does not exist."
    exit 1
fi

FILE_COUNT=$(docker exec "$SPARK_MASTER_ID" sh -c \
    'find "{PROCESSED_PATH}" -type f | wc -l')

echo "[INFO] Processed files: $FILE_COUNT"

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "[ERROR] No processed files found."
    exit 1
fi

echo "[OK] Processed data exists."

docker exec "$SPARK_MASTER_ID" sh -c \
    'find "{PROCESSED_PATH}" -maxdepth 2 -type f -print'

echo ""
echo "============================================================"
echo "TASK 5 COMPLETED"
echo "============================================================"
""",
    )

    # ========================================================
    # TASK 6 - MODEL TRAINING
    # ========================================================

    run_model_training = BashOperator(
        task_id="run_model_training",
        bash_command=f"""
set -e

echo "============================================================"
echo "TASK 6 - MODEL TRAINING"
echo "============================================================"

SPARK_MASTER_ID=$(docker ps -q -f "name=^spark-master$")

if [ -z "$SPARK_MASTER_ID" ]; then
    echo "[ERROR] Spark master is not running."
    exit 1
fi

TRAIN_SCRIPT="{BASE_PATH}/src/models/train.py"

if ! docker exec "$SPARK_MASTER_ID" test -f "$TRAIN_SCRIPT"; then
    echo "[ERROR] Training script not found:"
    echo "$TRAIN_SCRIPT"
    exit 1
fi

echo "[OK] Training script exists."

docker exec \
    -e DAG_ID="{DAG_ID}" \
    -e BASE_PATH="{BASE_PATH}" \
    -e MLFLOW_URI="{MLFLOW_URI}" \
    "$SPARK_MASTER_ID" \
    /opt/spark/bin/spark-submit \
    --master spark://spark-master:7077 \
    --total-executor-cores 2 \
    --executor-cores 2 \
    --executor-memory 4g \
    --driver-memory 1g \
    --conf spark.driver.cores=1 \
    --conf spark.driver.maxResultSize=1g \
    --conf spark.executor.memoryOverhead=384 \
    --conf spark.sql.shuffle.partitions=16 \
    --conf spark.default.parallelism=12 \
    --conf spark.sql.adaptive.enabled=true \
    --conf spark.jars.ivy=/tmp/.ivy2 \
    --name "{DAG_ID}_training" \
    "$TRAIN_SCRIPT"

echo ""
echo "[OK] Model training completed."

echo "============================================================"
echo "TASK 6 COMPLETED"
echo "============================================================"
""",
    )

    # ========================================================
    # TASK 7 - DEPLOY FASTAPI
    # ========================================================

    deploy_fastapi = BashOperator(
        task_id="deploy_fastapi",
        bash_command=f"""
set -e

echo "============================================================"
echo "TASK 7 - DEPLOY FASTAPI"
echo "============================================================"

BASE_DIR="{BASE_PATH}"
DOCKERFILE="$BASE_DIR/docker/Dockerfile"

echo "[INFO] API host port     : {API_PORT}"
echo "[INFO] API internal port : {API_INTERNAL_PORT}"
echo ""

echo "[CHECK] Project directory..."

if [ ! -d "$BASE_DIR" ]; then
    echo "[ERROR] Project directory does not exist:"
    echo "$BASE_DIR"
    exit 1
fi

echo "[OK] Project directory exists."

echo ""
echo "[CHECK] Dockerfile..."

if [ ! -f "$DOCKERFILE" ]; then
    echo "[ERROR] Dockerfile not found:"
    echo "$DOCKERFILE"
    exit 1
fi

echo "[OK] Dockerfile exists."

echo ""
echo "[BUILD] Building FastAPI Docker image..."

docker build \
    -t "{API_IMAGE}" \
    -f "$DOCKERFILE" \
    "$BASE_DIR"

echo ""
echo "[OK] Docker image built."

echo "[CLEANUP] Removing existing API container if present..."

docker rm -f "{API_CONTAINER}" >/dev/null 2>&1 || true

echo ""
echo "[RUN] Starting FastAPI container..."

docker run -d \
    --name "{API_CONTAINER}" \
    -p {API_PORT}:{API_INTERNAL_PORT} \
    -e DAG_ID="{DAG_ID}" \
    -e BASE_PATH="{BASE_PATH}" \
    -e MODEL_PATH="{MODEL_PATH}" \
    -e MLFLOW_URI="{MLFLOW_URI}" \
    -v "$BASE_DIR:$BASE_DIR" \
    "{API_IMAGE}"

echo ""
echo "[OK] FastAPI container started."

echo ""
echo "[INFO] FastAPI available at:"
echo "http://localhost:{API_PORT}"
echo ""

echo "[WAIT] Waiting for API..."

for i in $(seq 1 30); do

    if docker ps -q -f "name=^{API_CONTAINER}$" | grep -q .; then

        if docker exec "{API_CONTAINER}" \
            python -c \
            "import urllib.request; urllib.request.urlopen('http://127.0.0.1:{API_INTERNAL_PORT}/health', timeout=3); print('HEALTHY')" \
            >/dev/null 2>&1; then

            echo "[OK] API health check passed."
            break
        fi
    fi

    if [ "$i" -eq 30 ]; then
        echo "[ERROR] API did not become healthy."
        echo ""
        echo "[LOGS] Last 100 lines from API container:"
        docker logs "{API_CONTAINER}" --tail 100 || true
        exit 1
    fi

    sleep 2
done

echo ""
echo "============================================================"
echo "TASK 7 COMPLETED - FASTAPI DEPLOYED"
echo "============================================================"
""",
    )

    # ========================================================
    # TASK 8 - FINAL LOGS / STATUS
    # ========================================================

    fetch_logs = BashOperator(
        task_id="fetch_logs",
        trigger_rule="all_done",
        bash_command=f"""
echo "============================================================"
echo "TASK 8 - FINAL PIPELINE STATUS"
echo "============================================================"

CONTAINER="{API_CONTAINER}"
IMAGE="{API_IMAGE}"

echo ""
echo "--------------- API CONFIGURATION ----------------------"

echo "Host port     : {API_PORT}"
echo "Container port: {API_INTERNAL_PORT}"
echo "API URL       : http://localhost:{API_PORT}"
echo ""

echo "--------------- DOCKER CONTAINER STATUS ----------------"

docker ps -a -f "name=^/$CONTAINER$" || true

echo ""
echo "--------------- DOCKER IMAGE STATUS --------------------"

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[OK] Docker image exists: $IMAGE"
else
    echo "[WARNING] Docker image does not exist: $IMAGE"
fi

echo ""
echo "--------------- API CONTAINER LOGS ---------------------"

docker logs "$CONTAINER" --tail 50 2>&1 || true

echo ""
echo "--------------- API HEALTH CHECK ------------------------"

if docker ps -q -f "name=^$CONTAINER$" | grep -q .; then

    echo "[CHECK] Calling /health..."

    docker exec "$CONTAINER" \
        python -c \
        "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:{API_INTERNAL_PORT}/health', timeout=5).read().decode())" \
        || true

else
    echo "[WARNING] API container is not running."
fi

echo ""
echo "--------------- FINAL FILE STRUCTURE -------------------"

find "{BASE_PATH}/data" \
    -maxdepth 3 \
    -type f \
    -print \
    2>/dev/null || true

echo ""
echo "============================================================"
echo "TASK 8 COMPLETED - FINAL STATUS COLLECTED"
echo "============================================================"
""",
    )

    # ========================================================
    # DEPENDENCIES
    # ========================================================

    initialize_pipeline >> [team32_kp, team32_kc] >> validate_raw_data >> run_preprocessing >> validate_processed_data >> run_model_training >> deploy_fastapi >> fetch_logs