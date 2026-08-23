import json
import os

import pandas as pd
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError


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

BROKER = os.getenv("KAFKA_BROKER", "kafka:29092")

TOPIC = f"{DAG_ID}_producer"

INPUT_CSV = os.path.join(
    BASE_PATH,
    "data",
    "raw",
    "fraudTrain.csv"
)


# =========================================================
# KAFKA TOPIC
# =========================================================

def setup_topic():
    print("==================================================")
    print("KAFKA TOPIC SETUP")
    print("==================================================")

    print(f"Base Path       : {BASE_PATH}")
    print(f"DAG ID          : {DAG_ID}")
    print(f"Kafka Broker    : {BROKER}")
    print(f"Kafka Topic     : {TOPIC}")
    print("Retention       : 1 hour")
    print("Partitions      : 1")
    print("Replication     : 1")
    print("")

    admin_client = None

    try:
        admin_client = KafkaAdminClient(
            bootstrap_servers=[BROKER]
        )

        topic = NewTopic(
            name=TOPIC,
            num_partitions=1,
            replication_factor=1,
            topic_configs={
                "retention.ms": "3600000"
            }
        )

        admin_client.create_topics(
            new_topics=[topic],
            validate_only=False
        )

        print(f"[OK] Kafka topic created: {TOPIC}")

    except TopicAlreadyExistsError:
        print(f"[OK] Kafka topic already exists: {TOPIC}")

    except Exception as e:
        print(f"[WARNING] Kafka topic setup issue: {e}")

    finally:
        if admin_client:
            try:
                admin_client.close()
            except Exception:
                pass


# =========================================================
# PRODUCER
# =========================================================

def run_producer():

    print("")
    print("==================================================")
    print("TASK - KAFKA PRODUCER STARTED")
    print("==================================================")

    print(f"Base Path    : {BASE_PATH}")
    print(f"DAG ID       : {DAG_ID}")
    print(f"Kafka Broker : {BROKER}")
    print(f"Kafka Topic  : {TOPIC}")
    print(f"Input CSV    : {INPUT_CSV}")
    print("")

    # -----------------------------------------------------
    # Check input
    # -----------------------------------------------------

    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(
            f"Input CSV does not exist: {INPUT_CSV}"
        )

    file_size_mb = os.path.getsize(INPUT_CSV) / (1024 * 1024)

    print(f"[OK] Input file exists.")
    print(f"[INFO] Input file size: {file_size_mb:.2f} MB")
    print("")

    # -----------------------------------------------------
    # Create topic
    # -----------------------------------------------------

    setup_topic()

    # -----------------------------------------------------
    # Kafka producer
    # -----------------------------------------------------

    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda value:
            json.dumps(value).encode("utf-8"),
        key_serializer=lambda key:
            str(key).encode("utf-8"),
    )

    # -----------------------------------------------------
    # Read data
    # -----------------------------------------------------

    print("")
    print("[STEP] Loading CSV...")

    df = pd.read_csv(INPUT_CSV)

    total_records = len(df)

    print(f"[OK] CSV loaded successfully.")
    print(f"[INFO] Total records: {total_records}")
    print("")

    # -----------------------------------------------------
    # Produce messages
    # -----------------------------------------------------

    sent_count = 0

    print("[STEP] Publishing messages to Kafka...")

    for index, row in df.iterrows():

        transaction = row.to_dict()

        if "trans_date_trans_time" in transaction:
            transaction["trans_date_trans_time"] = str(
                transaction["trans_date_trans_time"]
            )

        trans_id = transaction.get(
            "trans_num",
            str(index)
        )

        producer.send(
            TOPIC,
            key=trans_id,
            value=transaction
        )

        sent_count += 1

        if sent_count % 100000 == 0:
            percentage = (
                sent_count / total_records
            ) * 100

            print(
                f"[PRODUCER PROGRESS] "
                f"{sent_count:,}/{total_records:,} "
                f"({percentage:.2f}%) messages sent"
            )

    # -----------------------------------------------------
    # Flush
    # -----------------------------------------------------

    print("")
    print("[STEP] Flushing Kafka producer...")

    producer.flush()
    producer.close()

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("")
    print("==================================================")
    print("KAFKA PRODUCER COMPLETED")
    print("==================================================")

    print(f"Kafka Topic        : {TOPIC}")
    print(f"CSV Records       : {total_records:,}")
    print(f"Messages Sent     : {sent_count:,}")
    print(f"Messages Remaining: {total_records - sent_count:,}")

    if sent_count != total_records:
        raise RuntimeError(
            "Producer count mismatch."
        )

    print("[SUCCESS] All records were published.")
    print("==================================================")


if __name__ == "__main__":
    run_producer()