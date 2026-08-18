import json
import os
import pandas as pd
from kafka import KafkaConsumer

# Server Infrastructure Configuration
BROKER = "kafka:29092"
TOPIC = "team32-fd-kafka"

# Writing to /tmp/ to avoid server permission errors
OUTPUT_CSV = "/storage/scratch/team32_consumed_transactions.csv"

def run_consumer(timeout_ms: int = 10000):
    print(f"==================================================")
    print(f"   Starting Kafka Consumer for Topic: {TOPIC}")
    print(f"==================================================")

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=[BROKER],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id="team32-fd-consumer-group-demo",
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        key_deserializer=lambda x: x.decode('utf-8') if x else '',
        consumer_timeout_ms=timeout_ms  # Stops listening automatically when stream is empty
    )

    consumed_records = []
    print(f"Listening on topic '{TOPIC}'... (will exit after {timeout_ms/1000} seconds of silence)")

    try:
        for message in consumer:
            # Append all messages from this dedicated topic
            consumed_records.append(message.value)
    except Exception as e:
        print(f"Consumer loop finished: {e}")
    finally:
        consumer.close()

    consumed_count = len(consumed_records)
    print(f"--> Consumed {consumed_count} records from '{TOPIC}'.")

    if consumed_count > 0:
        os.makedirs(os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else '.', exist_ok=True)
        pd.DataFrame(consumed_records).to_csv(OUTPUT_CSV, index=False)
        print(f"\n[Success] Consumed batch saved to {OUTPUT_CSV}")
    else:
        print("\n[Warning] No records were found or saved.")
        
    print(f"==================================================")

if __name__ == "__main__":
    run_consumer()