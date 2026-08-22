from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id='student_da25m512_fraud_detection_test_10',
    description="Kafka consumer for Team32 FraudDetection",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    is_paused_upon_creation=False,
    tags=['student-ui', 'kafka', 'fraud-detection', 'user:da25m512'],
    default_args={"owner": 'da25m512', "retries": 0},
) as dag:

# ---------------------------------------------------------
    # TASK 1: Data Ingestion (Kafka Producer)
    # ---------------------------------------------------------
    # Simulates live credit card swipes by sending data to the topic
    # run_producer = BashOperator(
    #     task_id="team32_kp",
    #     bash_command='docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --total-executor-cores 2 --executor-cores 2 --executor-memory 4g --driver-memory 1g --conf spark.driver.cores=1 --conf spark.driver.maxResultSize=1g --conf spark.executor.memoryOverhead=384 --conf spark.memory.fraction=0.600 --conf spark.memory.storageFraction=0.500 --conf spark.sql.shuffle.partitions=16 --conf spark.default.parallelism=12 --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128MB --conf spark.sql.adaptive.enabled=true --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 --name team32_Kafka_consumer_19thAug-da25m530 /storage/uploads/Team32_FraudDetection/src/data_ingestion/kafka_producer.py')
	
    # # ---------------------------------------------------------
    # # TASK 2: Data Extraction (Kafka Consumer)
    # # ---------------------------------------------------------
    # # Pulls the streamed data from the topic and saves it to a CSV
    # run_consumer = BashOperator(
    #     task_id="team32_kc",
    #     bash_command='docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --total-executor-cores 2 --executor-cores 2 --executor-memory 4g --driver-memory 1g --conf spark.driver.cores=1 --conf spark.driver.maxResultSize=1g --conf spark.executor.memoryOverhead=384 --conf spark.memory.fraction=0.600 --conf spark.memory.storageFraction=0.500 --conf spark.sql.shuffle.partitions=16 --conf spark.default.parallelism=12 --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128MB --conf spark.sql.adaptive.enabled=true --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 --name team32_Kafka_consumer_19thAug-da25m530 /storage/uploads/Team32_FraudDetection/src/data_ingestion/kafka_consumer.py')

    # ---------------------------------------------------------
    # TASK 3: Feature Engineering & Preprocessing
    # ---------------------------------------------------------
    # Cleans data, applies SMOTE, scales features, and saves the processed dataset
  
    # run_preprocessing = BashOperator(
    #      task_id='run_preprocessing',
    #      bash_command='docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --total-executor-cores 2 --executor-cores 2 --executor-memory 4g --driver-memory 1g --conf spark.driver.cores=1 --conf spark.driver.maxResultSize=1g /storage/uploads/Team32_FraudDetection/src/features/preprocessSpark.py'
    # )
    
    # ---------------------------------------------------------
    # TASK 4: Model Training & MLflow Logging
    # ---------------------------------------------------------
    # # Trains the model on the processed data and logs metrics to MLflow
#    run_training = BashOperator(
#         task_id='run_model_training',
#         bash_command=(
#             'docker exec '
#             '-e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" '
#             '-e MLFLOW_EXPERIMENT_NAME=team32_fraud_detection '
#             'spark-master /opt/spark/bin/spark-submit '
#             '--master spark://spark-master:7077 '
#             '--total-executor-cores 2 '
#             '--executor-cores 2 '
#             '--executor-memory 4g '
#             '--driver-memory 1g '
#             '--conf spark.driver.cores=1 '
#             '--conf spark.driver.maxResultSize=1g '
#             '/storage/uploads/Team32_FraudDetection/src/models/train.py'
#         )
#     )
    # ---------------------------------------------------------
    # TASK 5: Teardown Existing API Container
    # ---------------------------------------------------------
    # Safely spin down the old container to free up the port
   

    # ---------------------------------------------------------
    # TASK 6: Build and Deploy New API Container
    # ---------------------------------------------------------
    
    # ---------------------------------------------------------
    # TASK 7: Test the API Endpoint via cURL
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # TASK 7: Test the API Endpoint via Python (No cURL needed)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # TASK 7: Test the API Endpoint (With Detailed Error Catching)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # TASK 7: Test the API Endpoint (Quote-Safe Version)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # TASK 7: Test the Single Endpoint (21 Features Payload)
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # TASK 7: Test the Single Endpoint (Genuine Transaction Payload)
    # ---------------------------------------------------------
    test_api = BashOperator(
        task_id='test_api_python',
        bash_command="""
        docker exec team32-fraud-api python -c '
import urllib.request
import urllib.error
import json

url = "http://127.0.0.1:8000/predict"

# Payload adjusted to simulate a typical genuine transaction
payload = {
    "features": {
        "trans_hour": 10.0,
        "trans_day_of_week": 2.0,
        "age": 42.0,
        "city_pop": 120000.0,
        "amt_scaled": -0.15,          # Much lower transaction amount
        "distance_km_scaled": 0.02,   # Very close to the user
        "gender_M": 0.0,
        "cat_grocery_pos": 1.0,       # Changed to a typical in-person grocery purchase
        "cat_entertainment": 0.0,
        "cat_gas_transport": 0.0,
        "cat_grocery_net": 0.0,
        "cat_shopping_net": 0.0,
        "cat_shopping_pos": 0.0,
        "cat_travel": 0.0,
        "cat_misc_pos": 0.0,
        "cat_health_fitness": 0.0,
        "cat_kids_pets": 0.0,
        "cat_home": 0.0,
        "cat_personal_care": 0.0,
        "cat_food_dining": 0.0,
        "cat_misc_net": 0.0
    }
}

req = urllib.request.Request(
    url, 
    data=json.dumps(payload).encode("utf-8"), 
    headers={"Content-Type": "application/json"}
)

try:
    response = urllib.request.urlopen(req)
    parsed_json = json.loads(response.read().decode("utf-8"))
    print("SUCCESS! API Response:")
    print(json.dumps(parsed_json, indent=4))
except urllib.error.HTTPError as e:
    error_msg = e.read().decode("utf-8")
    print("API REJECTED THE REQUEST (400 Bad Request).")
    print("EXACT REASON: " + error_msg)
except Exception as e:
    print("OTHER ERROR: " + str(e))
'
        """
    )
    # ---------------------------------------------------------
    # Define Task Dependencies (Execution Order)
    # ---------------------------------------------------------
    # Assuming run_training is your Phase 4 task
    test_api

    # ---------------------------------------------------------
    # Define Task Dependencies (Execution Order)
    # ---------------------------------------------------------
    #run_producer >> run_consumer >> run_preprocessing >> run_training
    