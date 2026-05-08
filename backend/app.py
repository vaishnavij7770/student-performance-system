from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from models.database import db
from routes.auth import auth_bp
from routes.admin import admin_bp
from routes.teacher import teacher_bp
from routes.student import student_bp
from routes.attendance import attendance_bp
import os
from dotenv import load_dotenv

load_dotenv()

def create_app():
    app = Flask(__name__)

    # Config
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-change-in-prod')
    app.config['JWT_SECRET_KEY'] = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key-change-in-prod')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        'DATABASE_URL',
        'mysql+pymysql://root:12345678@localhost/student_performance_system'
    )
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
    app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Extensions
    CORS(app, origins=['http://localhost:3000'], supports_credentials=True)
    db.init_app(app)
    JWTManager(app)

    # Blueprints
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(teacher_bp, url_prefix='/api/teacher')
    app.register_blueprint(student_bp, url_prefix='/api/student')
    app.register_blueprint(attendance_bp, url_prefix='/api/attendance')

    with app.app_context():
        db.create_all()
        seed_admin()

    return app


def seed_admin():
    from models.user import User
    if not User.query.filter_by(role='admin').first():
        admin = User(name='Admin', email='admin@school.com', role='admin', is_approved=True)
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print("✅ Default admin created: admin@school.com / admin123")


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
