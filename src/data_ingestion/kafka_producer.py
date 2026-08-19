import json
import pandas as pd
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

# Server Infrastructure Configuration
BROKER = "kafka:29092"
TOPIC = "team32-fd-kafka-19th-Aug_3"
INPUT_CSV = "/storage/uploads/Team32_FraudDetection/data/raw/fraudTrain.csv"

def setup_topic_with_retention():
    """Ensure the topic exists with a 1-hour retention policy."""
    print(f"Checking/Creating topic '{TOPIC}' with 1-hour retention...")
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=[BROKER])
        
        # 1 hour = 60 mins * 60 secs * 1000 ms = 3,600,000 ms
        topic_list = [
            NewTopic(
                name=TOPIC, 
                num_partitions=1, 
                replication_factor=1, 
                topic_configs={'retention.ms': '3600000'}
            )
        ]
        admin_client.create_topics(new_topics=topic_list, validate_only=False)
        print(f"Topic '{TOPIC}' created successfully with 1-hour retention.")
    except TopicAlreadyExistsError:
        print(f"Topic '{TOPIC}' already exists. (Ensure retention was set previously).")
    except Exception as e:
        print(f"Admin client warning: {e}")
    finally:
        try:
            admin_client.close()
        except:
            pass

def run_producer():
    setup_topic_with_retention()

    print(f"==================================================")
    print(f"   Starting Kafka Producer for Topic: {TOPIC}")
    print(f"==================================================")

    producer = KafkaProducer(
        bootstrap_servers=[BROKER],
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: str(k).encode('utf-8')
    )

    print(f"Loading data from {INPUT_CSV}...")
    try:
        df = pd.read_csv(INPUT_CSV)
        total_records = len(df)
        print(f"Successfully loaded {total_records} records.")
    except FileNotFoundError:
        print(f"[Error] File not found at {INPUT_CSV}")
        return

    print("Publishing messages to Kafka...")
    for index, row in df.iterrows():
        transaction = row.to_dict()
        
        # Ensure timestamps or objects are string-serializable for JSON
        if 'trans_date_trans_time' in transaction:
            transaction['trans_date_trans_time'] = str(transaction['trans_date_trans_time'])

        # Use transaction number as the message key if available, otherwise use index
        trans_id = transaction.get('trans_num', str(index))

        producer.send(TOPIC, key=trans_id, value=transaction)

        # Optional: print progress every 100,000 records
        if (index + 1) % 100000 == 0:
            print(f"Produced {index + 1} / {total_records} messages...")

    # Ensure all messages are pushed out before closing
    producer.flush()
    producer.close()
    
    print(f"--> Successfully produced {total_records} messages to '{TOPIC}'")
    print(f"==================================================")

if __name__ == "__main__":
    run_producer()