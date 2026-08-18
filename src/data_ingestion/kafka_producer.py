import pandas as pd
import json
import time
from kafka import KafkaProducer

def initialize_producer(bootstrap_servers=['localhost:9092']):
    """Initializes and returns a Kafka producer instance."""
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

def stream_dataset(file_path: str, topic_name: str = 'live_transactions', delay: float = 0.1):
    """
    Streams rows from the dataset to a Kafka topic as simulated live swipes.
    """
    print(f"Loading dataset from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Sort chronologically if time column exists
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df = df.sort_values('trans_date_trans_time')

    producer = initialize_producer()
    print(f"Starting transaction stream to topic: '{topic_name}'...")

    for index, row in df.iterrows():
        transaction = row.to_dict()
        
        # Convert timestamps or non-serializable objects to string
        if 'trans_date_trans_time' in transaction:
            transaction['trans_date_trans_time'] = str(transaction['trans_date_trans_time'])
            
        producer.send(topic_name, value=transaction)
        print(f"[Sent] Transaction ID: {transaction.get('trans_num', index)} | Amount: ${transaction.get('amt', 0.0)}")
        
        time.sleep(delay)

if __name__ == "__main__":
    stream_dataset('data/raw/fraud_train.csv')