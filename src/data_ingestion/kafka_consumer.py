import json
import os

import pandas as pd
from kafka import KafkaConsumer


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

BROKER = os.getenv(
    "KAFKA_BROKER",
    "kafka:29092"
)

TOPIC = f"{DAG_ID}_producer"

CONSUMER_GROUP = f"{DAG_ID}_consumer_group"

OUTPUT_CSV = os.path.join(
    BASE_PATH,
    "data",
    "raw",
    f"{DAG_ID}_kafka_consumed.csv"
)


# =========================================================
# CONSUMER
# =========================================================

def run_consumer(timeout_ms: int = 10000):

    print("")
    print("==================================================")
    print("TASK - KAFKA CONSUMER STARTED")
    print("==================================================")

    print(f"Base Path       : {BASE_PATH}")
    print(f"DAG ID          : {DAG_ID}")
    print(f"Kafka Broker    : {BROKER}")
    print(f"Kafka Topic     : {TOPIC}")
    print(f"Consumer Group  : {CONSUMER_GROUP}")
    print(f"Output CSV      : {OUTPUT_CSV}")
    print("")

    os.makedirs(
        os.path.dirname(OUTPUT_CSV),
        exist_ok=True
    )

    # -----------------------------------------------------
    # Create consumer
    # -----------------------------------------------------

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[BROKER],

        auto_offset_reset="earliest",

        enable_auto_commit=True,

        group_id=CONSUMER_GROUP,

        value_deserializer=lambda x:
            json.loads(x.decode("utf-8")),

        key_deserializer=lambda x:
            x.decode("utf-8") if x else "",

        consumer_timeout_ms=timeout_ms,
    )

    consumed_records = []

    print(
        f"[STEP] Listening to topic '{TOPIC}'..."
    )

    print(
        f"[INFO] Consumer stops after "
        f"{timeout_ms / 1000:.0f} seconds of silence."
    )

    print("")

    # -----------------------------------------------------
    # Consume
    # -----------------------------------------------------

    try:

        for message in consumer:

            consumed_records.append(
                message.value
            )

            count = len(consumed_records)

            if count % 100000 == 0:
                print(
                    f"[CONSUMER PROGRESS] "
                    f"{count:,} messages consumed"
                )

    except Exception as e:

        print(
            f"[WARNING] Consumer loop finished: {e}"
        )

    finally:

        consumer.close()

    # -----------------------------------------------------
    # Final count
    # -----------------------------------------------------

    consumed_count = len(consumed_records)

    print("")
    print("==================================================")
    print("KAFKA CONSUMPTION COMPLETED")
    print("==================================================")

    print(f"Kafka Topic       : {TOPIC}")
    print(f"Consumer Group    : {CONSUMER_GROUP}")
    print(f"Messages Consumed : {consumed_count:,}")

    if consumed_count == 0:

        print(
            "[ERROR] No Kafka messages were consumed."
        )

        raise RuntimeError(
            "Kafka consumer received zero messages."
        )

    # -----------------------------------------------------
    # Save directly to base path
    # -----------------------------------------------------

    print("")
    print("[STEP] Saving consumed data...")

    pd.DataFrame(
        consumed_records
    ).to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("")
    print(f"[SUCCESS] Output written to:")
    print(OUTPUT_CSV)

    print(
        f"[SUCCESS] Total records written: "
        f"{consumed_count:,}"
    )

    print("==================================================")


if __name__ == "__main__":
    run_consumer()