from .database import db
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

class User(db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(512), nullable=False)
    role = db.Column(db.Enum('admin', 'teacher', 'student'), nullable=False)
    is_approved = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    student_profile = db.relationship('Student', backref='user', uselist=False, lazy=True)
    teacher_profile = db.relationship('Teacher', backref='user', uselist=False, lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    # ✅ FIXED FUNCTION
    def check_password(self, password):
        import bcrypt

        # 1. If plain text (your old data)
        if self.password_hash == password:
            return True

        # 2. If Werkzeug hash
        try:
            if check_password_hash(self.password_hash, password):
                return True
        except:
            pass

        # 3. If bcrypt hash
        try:
            return bcrypt.checkpw(password.encode(), self.password_hash.encode())
        except:
            return False

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'role': self.role,
            'is_approved': self.is_approved,
            'created_at': self.created_at.isoformat()
        }