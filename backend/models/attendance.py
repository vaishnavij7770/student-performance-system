from .database import db
from datetime import datetime

class Attendance(db.Model):
    __tablename__ = 'attendance'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    semester = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date, nullable=False)
    status = db.Column(db.Enum('present', 'absent', 'late'), default='present')
    remarks = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    student = db.relationship('Student', backref='attendance_records', lazy=True)
    teacher = db.relationship('Teacher', backref='attendance_given', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'teacher_id': self.teacher_id,
            'subject': self.subject,
            'semester': self.semester,
            'date': self.date.isoformat(),
            'status': self.status,
            'remarks': self.remarks,
            'created_at': self.created_at.isoformat(),
            'student_name': self.student.user.name if self.student and self.student.user else None,
            'teacher_name': self.teacher.user.name if self.teacher and self.teacher.user else None,
        }
