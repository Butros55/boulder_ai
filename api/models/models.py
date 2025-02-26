from datetime import datetime
from extensions import db

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

class Analysis(db.Model):
    __tablename__ = "analyses"
    id = db.Column(db.Integer, primary_key=True)
    image_height = db.Column(db.Integer)
    image_width = db.Column(db.Integer)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    detections = db.Column(db.JSON, nullable=True)
    original_image = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
