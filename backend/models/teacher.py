from .database import db

class Teacher(db.Model):
    __tablename__ = 'teachers'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    employee_id = db.Column(db.String(50), unique=True, nullable=False)
    department = db.Column(db.String(100))
    subject_specialization = db.Column(db.String(150))
    phone = db.Column(db.String(20), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'employee_id': self.employee_id,
            'department': self.department,
            'subject_specialization': self.subject_specialization,
            'phone': self.phone,
            'name': self.user.name if self.user else None,
            'email': self.user.email if self.user else None,
            'is_approved': self.user.is_approved if self.user else False,
        }
