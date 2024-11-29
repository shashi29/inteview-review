import pika
import json
from typing import Dict, Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RabbitMQClient:
    def __init__(self, host: str, queue: str, user: str, password: str):
        self.host = host
        self.queue = queue
        self.user = user
        self.password = password

    def _get_connection(self):
        credentials = pika.PlainCredentials(self.user, self.password)
        return pika.BlockingConnection(pika.ConnectionParameters(host=self.host, credentials=credentials))

    def _declare_queue(self, channel):
        channel.queue_declare(queue=self.queue, durable=True)

    def send_message(self, message: Dict):
        try:
            with self._get_connection() as connection:
                channel = connection.channel()
                self._declare_queue(channel)
                channel.basic_publish(
                    exchange='',
                    routing_key=self.queue,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                    )
                )
            logger.info(f"Sent message: {message}")
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    def start_consumer(self, callback: Callable):
        def internal_callback(ch, method, properties, body):
            message = json.loads(body)
            logger.info(f"Received message: {message}")
            callback(message)
            ch.basic_ack(delivery_tag=method.delivery_tag)

        try:
            with self._get_connection() as connection:
                channel = connection.channel()
                self._declare_queue(channel)
                channel.basic_qos(prefetch_count=1)
                channel.basic_consume(queue=self.queue, on_message_callback=internal_callback)
                logger.info('Waiting for messages. To exit press CTRL+C')
                channel.start_consuming()
        except Exception as e:
            logger.error(f"Error in consumer: {e}")

    def check_health(self):
        try:
            with self._get_connection() as connection:
                channel = connection.channel()
                self._declare_queue(channel)
            return True  # If successful, the service is healthy
        except Exception as e:
            logger.error(f"RabbitMQ health check failed: {e}")
            return False


    def publish_message(self, message: Dict):
        """
        Publish a message to the configured RabbitMQ queue.
        
        Args:
            message (Dict): The message to be published as a dictionary
        
        Returns:
            bool: True if message was successfully published, False otherwise
        """
        try:
            with self._get_connection() as connection:
                channel = connection.channel()
                self._declare_queue(channel)
                
                channel.basic_publish(
                    exchange='',
                    routing_key=self.queue,
                    body=json.dumps(message),
                    properties=pika.BasicProperties(
                        delivery_mode=2,  # make message persistent
                    )
                )
            
            logger.info(f"Published message successfully: {message}")
            return True
        
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False