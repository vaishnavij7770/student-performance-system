from .database import db
from datetime import datetime

class Mark(db.Model):
    __tablename__ = 'marks'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    teacher_id = db.Column(db.Integer, db.ForeignKey('teachers.id'), nullable=False)
    subject = db.Column(db.String(100), nullable=False)
    semester = db.Column(db.Integer, nullable=False)
    marks_obtained = db.Column(db.Float, nullable=False)
    total_marks = db.Column(db.Float, nullable=False, default=100)
    exam_type = db.Column(db.Enum('midterm', 'final', 'assignment', 'quiz'), default='final')
    remarks = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    teacher = db.relationship('Teacher', backref='marks_given', lazy=True)

    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'teacher_id': self.teacher_id,
            'subject': self.subject,
            'semester': self.semester,
            'marks_obtained': self.marks_obtained,
            'total_marks': self.total_marks,
            'percentage': round((self.marks_obtained / self.total_marks) * 100, 2),
            'exam_type': self.exam_type,
            'remarks': self.remarks,
            'created_at': self.created_at.isoformat(),
            'teacher_name': self.teacher.user.name if self.teacher and self.teacher.user else None,
        }
