import os
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from extensions import db, jwt

load_dotenv()

app = Flask(__name__)
CORS(app)

app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SQLALCHEMY_DATABASE_URI")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)
jwt.init_app(app)

from routes.auth import auth_bp
from routes.analysis import analysis_bp
from routes.yolo_endpoint import yolo_bp

app.register_blueprint(auth_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(yolo_bp)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    port = int(os.getenv('SERVER_PORT', '5000'))
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG') == '1', use_reloader=False)
