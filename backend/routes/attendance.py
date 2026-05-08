from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.database import db
from models.user import User
from models.student import Student
from models.teacher import Teacher
from models.attendance import Attendance
from datetime import date, datetime
from sqlalchemy import func

attendance_bp = Blueprint('attendance', __name__)

def get_teacher(user_id):
    user = User.query.get(user_id)
    if not user or user.role != 'teacher':
        return None
    return user.teacher_profile


# POST /api/attendance/bulk  — mark attendance for multiple students at once
@attendance_bp.route('/bulk', methods=['POST'])
@jwt_required()
def mark_bulk():
    user_id = int(get_jwt_identity())
    teacher = get_teacher(user_id)
    if not teacher:
        return jsonify({'error': 'Teacher access required'}), 403

    data = request.get_json()
    # data = { subject, semester, date, records: [{student_id, status, remarks}] }
    required = ['subject', 'semester', 'date', 'records']
    for f in required:
        if not data.get(f):
            return jsonify({'error': f'{f} is required'}), 400

    att_date = date.fromisoformat(data['date'])
    added = 0
    for rec in data['records']:
        # Upsert: delete existing for same student+subject+date then insert
        Attendance.query.filter_by(
            student_id=rec['student_id'],
            subject=data['subject'],
            date=att_date
        ).delete()
        att = Attendance(
            student_id=rec['student_id'],
            teacher_id=teacher.id,
            subject=data['subject'],
            semester=int(data['semester']),
            date=att_date,
            status=rec.get('status', 'present'),
            remarks=rec.get('remarks')
        )
        db.session.add(att)
        added += 1

    db.session.commit()
    return jsonify({'message': f'Attendance marked for {added} students'}), 201


# GET /api/attendance/student/<id>?semester=&subject=
@attendance_bp.route('/student/<int:student_id>', methods=['GET'])
@jwt_required()
def get_student_attendance(student_id):
    user_id = int(get_jwt_identity())
    # Allow teacher OR the student themselves
    user = User.query.get(user_id)
    if user.role == 'student':
        if not user.student_profile or user.student_profile.id != student_id:
            return jsonify({'error': 'Unauthorized'}), 403
    elif user.role == 'teacher':
        pass  # teachers can view any student
    else:
        return jsonify({'error': 'Unauthorized'}), 403

    semester = request.args.get('semester', type=int)
    subject = request.args.get('subject')

    query = Attendance.query.filter_by(student_id=student_id)
    if semester:
        query = query.filter_by(semester=semester)
    if subject:
        query = query.filter_by(subject=subject)

    records = query.order_by(Attendance.date.desc()).all()

    # Build summary per subject
    summary = {}
    for r in records:
        key = r.subject
        if key not in summary:
            summary[key] = {'present': 0, 'absent': 0, 'late': 0, 'total': 0}
        summary[key][r.status] += 1
        summary[key]['total'] += 1

    for subj in summary:
        total = summary[subj]['total']
        present = summary[subj]['present'] + summary[subj]['late']
        summary[subj]['percentage'] = round((present / total) * 100, 1) if total else 0

    return jsonify({
        'records': [r.to_dict() for r in records],
        'summary': summary
    }), 200


# GET /api/attendance/report?semester=&date=  (teacher)
@attendance_bp.route('/report', methods=['GET'])
@jwt_required()
def get_report():
    user_id = int(get_jwt_identity())
    teacher = get_teacher(user_id)
    if not teacher:
        return jsonify({'error': 'Teacher access required'}), 403

    semester = request.args.get('semester', type=int)
    att_date = request.args.get('date')
    subject = request.args.get('subject')

    query = Attendance.query.filter_by(teacher_id=teacher.id)
    if semester:
        query = query.filter_by(semester=semester)
    if att_date:
        query = query.filter_by(date=date.fromisoformat(att_date))
    if subject:
        query = query.filter_by(subject=subject)

    records = query.order_by(Attendance.date.desc()).all()
    return jsonify([r.to_dict() for r in records]), 200
