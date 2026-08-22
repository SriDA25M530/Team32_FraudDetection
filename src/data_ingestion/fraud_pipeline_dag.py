from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule

with DAG(
    dag_id='team32_final_end_to_end_pipeline_V1',
    description="Full End-to-End MLOps Pipeline for Team32 FraudDetection",
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
    run_producer = BashOperator(
        task_id="team32_kp",
        bash_command='docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --total-executor-cores 2 --executor-cores 2 --executor-memory 4g --driver-memory 1g --conf spark.driver.cores=1 --conf spark.driver.maxResultSize=1g --conf spark.executor.memoryOverhead=384 --conf spark.memory.fraction=0.600 --conf spark.memory.storageFraction=0.500 --conf spark.sql.shuffle.partitions=16 --conf spark.default.parallelism=12 --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128MB --conf spark.sql.adaptive.enabled=true --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 --name team32_Kafka_producer-da25m530 /storage/uploads/Team32_FraudDetection/src/data_ingestion/kafka_producer.py'
    )
    
    # ---------------------------------------------------------
    # TASK 2: Data Extraction (Kafka Consumer)
    # ---------------------------------------------------------
    run_consumer = BashOperator(
        task_id="team32_kc",
        bash_command='docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --total-executor-cores 2 --executor-cores 2 --executor-memory 4g --driver-memory 1g --conf spark.driver.cores=1 --conf spark.driver.maxResultSize=1g --conf spark.executor.memoryOverhead=384 --conf spark.memory.fraction=0.600 --conf spark.memory.storageFraction=0.500 --conf spark.sql.shuffle.partitions=16 --conf spark.default.parallelism=12 --conf spark.sql.adaptive.advisoryPartitionSizeInBytes=128MB --conf spark.sql.adaptive.enabled=true --conf spark.jars.ivy=/tmp/.ivy2 --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0 --name team32_Kafka_consumer-da25m530 /storage/uploads/Team32_FraudDetection/src/data_ingestion/kafka_consumer.py'
    )

    # ---------------------------------------------------------
    # TASK 3: Copy Raw Data
    # ---------------------------------------------------------
    copy_raw_data = BashOperator(
        task_id='copy_raw_data',
        bash_command='''
        mkdir -p /storage/uploads/Team32_FraudDetection/data/raw
        cp /storage/scratch/team32_fd_kafka_consumed.csv /storage/uploads/Team32_FraudDetection/data/raw/
        '''
    )

    # ---------------------------------------------------------
    # TASK 4: Feature Engineering & Preprocessing
    # ---------------------------------------------------------
    run_preprocessing = BashOperator(
        task_id='run_preprocessing',
        bash_command='docker exec spark-master /opt/spark/bin/spark-submit --master spark://spark-master:7077 --total-executor-cores 2 --executor-cores 2 --executor-memory 4g --driver-memory 1g --conf spark.driver.cores=1 --conf spark.driver.maxResultSize=1g /storage/uploads/Team32_FraudDetection/src/features/preprocessSpark.py'
    )
    
    # ---------------------------------------------------------
    # TASK 5: Copy Processed Data
    # ---------------------------------------------------------
    copy_processed_data = BashOperator(
        task_id='copy_processed_data',
        bash_command='''
        mkdir -p /storage/uploads/Team32_FraudDetection/data/processed
        rm -rf /storage/uploads/Team32_FraudDetection/data/processed/*
        cp -r /storage/scratch/team32_processed_transactions_spark/* /storage/uploads/Team32_FraudDetection/data/processed/
        '''
    )

    # ---------------------------------------------------------
    # TASK 6: Model Training & MLflow Logging
    # ---------------------------------------------------------
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
    # TASK 7: Copy Trained Models
    # ---------------------------------------------------------
    copy_models = BashOperator(
        task_id='copy_models',
        bash_command='''
        mkdir -p /storage/uploads/Team32_FraudDetection/data/models/Team32_FraudDetection
        rm -rf /storage/uploads/Team32_FraudDetection/data/models/Team32_FraudDetection/*
        cp -r /storage/scratch/Team32_FraudDetection/logistic_regression_model /storage/uploads/Team32_FraudDetection/data/models/Team32_FraudDetection/
        cp -r /storage/scratch/Team32_FraudDetection/gbt_classifier_model /storage/uploads/Team32_FraudDetection/data/models/Team32_FraudDetection/
        '''
    )

 
    

    # ---------------------------------------------------------
    # TASK 8: Build and Deploy New API Container
    # ---------------------------------------------------------
    deploy_api = BashOperator(
        task_id='deploy_fastapi',
        bash_command=(
            'cd /storage/uploads/Team32_FraudDetection && '
            'docker rm -f team32-fraud-api || true && '
            'docker build -t team32-fraud-api -f docker/Dockerfile . && '
            'docker run -d --name team32-fraud-api -p 8000:8000 -v /storage:/storage -e PYTHONUNBUFFERED=1 team32-fraud-api'
        )
    )

    # ---------------------------------------------------------
    # TASK 9: Fetch Docker Logs (Runs Even if Deploy Fails)
    # ---------------------------------------------------------
    fetch_docker_logs = BashOperator(
        task_id="fetch_logs",
        bash_command="docker logs team32-fraud-api --tail 50 ",
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ---------------------------------------------------------
    # Define Task Dependencies (Execution Order)
    # ---------------------------------------------------------
    run_producer >> run_consumer >> copy_raw_data >> run_preprocessing >> \
    copy_processed_data >> run_training >> copy_models >> deploy_api >> fetch_docker_logs