from confluent_kafka import Producer, Consumer, KafkaError
import json
import time

class KafkaManager:
    def __init__(self, bootstrap_servers='host.docker.internal:9092'):
        self.bootstrap_servers = bootstrap_servers
    
    def create_producer(self):
        """Create Kafka Producer"""
        return Producer({
            'bootstrap.servers': self.bootstrap_servers
        })
    
    def create_consumer(self, group_id, topics):
        """Create Kafka Consumer"""
        return Consumer({
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'earliest'
        })
    
    def produce_message(self, topic, message):
        """Produce message to Kafka topic"""
        producer = self.create_producer()
        
        def delivery_report(err, msg):
            if err is not None:
                print(f'Message delivery failed: {err}')
            else:
                print(f'Message delivered to {msg.topic()} [{msg.partition()}]')
        
        producer.produce(topic, json.dumps(message).encode('utf-8'), callback=delivery_report)
        producer.flush()
    
    def consume_messages(self, topic, timeout=10):
        """Consume messages from Kafka topic"""
        consumer = self.create_consumer('test-group', [topic])
        consumer.subscribe([topic])
        
        messages = []
        start_time = time.time()
        
        try:
            while time.time() - start_time < timeout:
                msg = consumer.poll(1.0)
                
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        print('Reached end of partition')
                    else:
                        print(f'Error: {msg.error()}')
                    continue
                
                message = json.loads(msg.value().decode('utf-8'))
                messages.append(message)
                print(f'Received message: {message}')
        
        except Exception as e:
            print(f'Error consuming messages: {e}')
        
        finally:
            consumer.close()
            return messages

def test_kafka_operations():
    """Test Kafka operations"""
    kafka_manager = KafkaManager()
    test_topic = 'test_topic'
    
    # Produce test messages
    test_messages = [
        {'id': 1, 'message': 'First test message'},
        {'id': 2, 'message': 'Second test message'}
    ]
    
    for message in test_messages:
        kafka_manager.produce_message(test_topic, message)
    
    # Consume messages
    received_messages = kafka_manager.consume_messages(test_topic)
    
    # Validate results
    assert len(received_messages) > 0, "No messages received"
    print("Kafka test completed successfully!")

if __name__ == '__main__':
    test_kafka_operations()

# Requirements:
# pip install confluent-kafka