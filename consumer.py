from confluent_kafka import Consumer, KafkaError
import json
from database import SessionLocal, EquipmentTelemetry, init_db
import config
import time

# Initialize database
init_db()

# Consumer configuration
kafka_config = {
    'bootstrap.servers': config.KAFKA_SERVER,
    'group.id': 'eaglevision-group',
    'auto.offset.reset': 'earliest'
}

def consume():
    consumer = Consumer(kafka_config)
    consumer.subscribe([config.KAFKA_TOPIC])
    
    print("Consumer started. Waiting for messages...")
    
    try:
        while True:
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                elif msg.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    time.sleep(1)
                    continue
                else:
                    print(f"Consumer error: {msg.error()}")
                    break

            # Process the message
            try:
                data = json.loads(msg.value().decode('utf-8'))
                print(f"Received: {data}")
                
                # Save to database
                db = SessionLocal()
                telemetry = EquipmentTelemetry(
                    frame_id=data.get("frame_id"),
                    equipment_id=data.get("equipment_id"),
                    equipment_class=data.get("equipment_class"),
                    vid_timestamp=data.get("timestamp"),
                    current_state=data.get("utilization", {}).get("current_state"),
                    current_activity=data.get("utilization", {}).get("current_activity"),
                    motion_source=data.get("utilization", {}).get("motion_source"),
                    total_tracked_seconds=data.get("time_analytics", {}).get("total_tracked_seconds"),
                    total_active_seconds=data.get("time_analytics", {}).get("total_active_seconds"),
                    total_idle_seconds=data.get("time_analytics", {}).get("total_idle_seconds"),
                    utilization_percent=data.get("time_analytics", {}).get("utilization_percent")
                )
                db.add(telemetry)
                db.commit()
                db.close()
                
            except Exception as e:
                print(f"Error processing message: {e}")
                
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()

if __name__ == "__main__":
    consume()
