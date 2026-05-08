from .database import db

class Student(db.Model):
    __tablename__ = 'students'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    roll_number = db.Column(db.String(50), unique=True, nullable=False)
    department = db.Column(db.String(100))
    current_semester = db.Column(db.Integer, default=1)
    photo_url = db.Column(db.String(255), nullable=True)
    date_of_birth = db.Column(db.Date, nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    address = db.Column(db.Text, nullable=True)

    marks = db.relationship('Mark', backref='student', lazy=True)
    activities = db.relationship('Activity', backref='student', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'roll_number': self.roll_number,
            'department': self.department,
            'current_semester': self.current_semester,
            'photo_url': self.photo_url,
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'phone': self.phone,
            'address': self.address,
            'name': self.user.name if self.user else None,
            'email': self.user.email if self.user else None,
            'is_approved': self.user.is_approved if self.user else False,
        }
