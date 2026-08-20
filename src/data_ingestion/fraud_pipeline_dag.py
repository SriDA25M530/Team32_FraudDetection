from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

with DAG(
    dag_id='student_da25m512_fraud_detection_consumer_19th_Aug_8',
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
   run_training = BashOperator(
        task_id='run_model_training',
        bash_command=(
            'docker exec '
            '-e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" '
            '-e MLFLOW_EXPERIMENT_NAME=team32_fraud_detection '
            'spark-master /opt/spark/bin/spark-submit '
            '--master spark://spark-master:7077 '
            '--total-executor-cores 2 '
            '--executor-cores 2 '
            '--executor-memory 4g '
            '--driver-memory 1g '
            '--conf spark.driver.cores=1 '
            '--conf spark.driver.maxResultSize=1g '
            '/storage/uploads/Team32_FraudDetection/src/models/train.py'
        )
    )

    # ---------------------------------------------------------
    # Define Task Dependencies (Execution Order)
    # ---------------------------------------------------------
    #run_producer >> run_consumer >> run_preprocessing >> run_training
    run_training