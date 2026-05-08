from .database import db
from datetime import datetime

class Activity(db.Model):
    __tablename__ = 'activities'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    category = db.Column(db.Enum('sports', 'cultural', 'technical', 'social', 'academic', 'other'), default='other')
    description = db.Column(db.Text, nullable=True)
    semester = db.Column(db.Integer, nullable=False)
    achievement = db.Column(db.String(100), nullable=True)
    date = db.Column(db.Date, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    teacher = db.relationship('Teacher', backref='activities_added', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'teacher_id': self.teacher_id,
            'title': self.title,
            'category': self.category,
            'description': self.description,
            'semester': self.semester,
            'achievement': self.achievement,
            'date': self.date.isoformat() if self.date else None,
            'created_at': self.created_at.isoformat(),
            'teacher_name': self.teacher.user.name if self.teacher and self.teacher.user else None,
        }
