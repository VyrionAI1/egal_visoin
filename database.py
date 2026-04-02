from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import config

Base = declarative_base()

class EquipmentTelemetry(Base):
    __tablename__ = 'equipment_telemetry'
    
    id = Column(Integer, primary_key=True)
    frame_id = Column(Integer)
    equipment_id = Column(String)
    equipment_class = Column(String)
    vid_timestamp = Column(String)
    current_state = Column(String)
    current_activity = Column(String)
    motion_source = Column(String)
    total_tracked_seconds = Column(Float)
    total_active_seconds = Column(Float)
    total_idle_seconds = Column(Float)
    utilization_percent = Column(Float)
    db_timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# Database URL from config
DATABASE_URL = config.DATABASE_URL

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    init_db()
    print("Database initialized.")
