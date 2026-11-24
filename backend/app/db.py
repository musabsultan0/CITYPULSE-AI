from pathlib import Path
from datetime import datetime, timezone
from sqlalchemy import create_engine, String, DateTime, Float, Text, JSON
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Mapped, mapped_column
from backend.app.core.config import settings, ROOT_DIR

db_url = settings.database_url
if db_url.startswith("sqlite:///./"):
    rel_path = db_url.removeprefix("sqlite:///./")
    resolved = (ROOT_DIR / rel_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    db_url = f"sqlite:///{resolved.as_posix()}"
    connect_args = {"check_same_thread": False}
elif db_url.startswith("sqlite"):
    (ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)
    connect_args = {"check_same_thread": False}
else:
    connect_args = {}
engine = create_engine(db_url, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

class Incident(Base):
    __tablename__ = "incidents"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    zone: Mapped[str] = mapped_column(String(100), index=True)
    scenario: Mapped[str] = mapped_column(String(50))
    severity: Mapped[str] = mapped_column(String(20))
    risk_score: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30), default="awaiting_review")
    summary: Mapped[str] = mapped_column(Text)
    result_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

class Approval(Base):
    __tablename__ = "approvals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    incident_id: Mapped[str] = mapped_column(String(36), index=True)
    reviewer: Mapped[str] = mapped_column(String(100))
    decision: Mapped[str] = mapped_column(String(20))
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
