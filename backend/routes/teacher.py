from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.database import db
from models.user import User
from models.student import Student
from models.teacher import Teacher
from models.mark import Mark
from models.activity import Activity
from datetime import date

teacher_bp = Blueprint('teacher', __name__)

def get_teacher(user_id):
    user = User.query.get(user_id)
    if not user or user.role != 'teacher':
        return None
    return user.teacher_profile


@teacher_bp.route('/students', methods=['GET'])
@jwt_required()
def get_students():
    user_id = int(get_jwt_identity())
    teacher = get_teacher(user_id)
    if not teacher:
        return jsonify({'error': 'Teacher access required'}), 403

    semester = request.args.get('semester', type=int)
    query = Student.query.join(User).filter(User.is_approved == True)
    if semester:
        query = query.filter(Student.current_semester == semester)
    students = query.order_by(Student.current_semester).all()
    return jsonify([s.to_dict() for s in students]), 200


@teacher_bp.route('/marks', methods=['POST'])
@jwt_required()
def add_mark():
    user_id = int(get_jwt_identity())
    teacher = get_teacher(user_id)
    if not teacher:
        return jsonify({'error': 'Teacher access required'}), 403

    data = request.get_json()
    required = ['student_id', 'subject', 'semester', 'marks_obtained', 'total_marks']
    for f in required:
        if data.get(f) is None:
            return jsonify({'error': f'{f} is required'}), 400

    mark = Mark(
        student_id=data['student_id'],
        teacher_id=teacher.id,
        subject=data['subject'],
        semester=data['semester'],
        marks_obtained=data['marks_obtained'],
        total_marks=data['total_marks'],
        exam_type=data.get('exam_type', 'final'),
        remarks=data.get('remarks')
    )
    db.session.add(mark)
    db.session.commit()
    return jsonify({'message': 'Mark added', 'mark': mark.to_dict()}), 201


@teacher_bp.route('/marks/<int:student_id>', methods=['GET'])
@jwt_required()
def get_student_marks(student_id):
    user_id = int(get_jwt_identity())
    teacher = get_teacher(user_id)
    if not teacher:
        return jsonify({'error': 'Teacher access required'}), 403

    semester = request.args.get('semester', type=int)
    query = Mark.query.filter_by(student_id=student_id)
    if semester:
        query = query.filter_by(semester=semester)
    marks = query.all()
    return jsonify([m.to_dict() for m in marks]), 200


@teacher_bp.route('/marks/<int:mark_id>', methods=['PUT'])
@jwt_required()
def update_mark(mark_id):
    user_id = int(get_jwt_identity())
    teacher = get_teacher(user_id)
    if not teacher:
        return jsonify({'error': 'Teacher access required'}), 403

    mark = Mark.query.get_or_404(mark_id)
    if mark.teacher_id != teacher.id:
        return jsonify({'error': 'Unauthorized'}), 403

    data = request.get_json()
    for field in ['subject', 'semester', 'marks_obtained', 'total_marks', 'exam_type', 'remarks']:
        if field in data:
            setattr(mark, field, data[field])
    db.session.commit()
    return jsonify({'message': 'Mark updated', 'mark': mark.to_dict()}), 200


@teacher_bp.route('/marks/<int:mark_id>', methods=['DELETE'])
@jwt_required()
def delete_mark(mark_id):
    user_id = int(get_jwt_identity())
    teacher = get_teacher(user_id)
    if not teacher:
        return jsonify({'error': 'Teacher access required'}), 403

    mark = Mark.query.get_or_404(mark_id)
    if mark.teacher_id != teacher.id:
        return jsonify({'error': 'Unauthorized'}), 403

    db.session.delete(mark)
    db.session.commit()
    return jsonify({'message': 'Mark deleted'}), 200


@teacher_bp.route('/activities', methods=['POST'])
@jwt_required()
def add_activity():
    user_id = int(get_jwt_identity())
    teacher = get_teacher(user_id)
    if not teacher:
        return jsonify({'error': 'Teacher access required'}), 403

    data = request.get_json()
    activity = Activity(
        student_id=data['student_id'],
        teacher_id=teacher.id,
        title=data['title'],
        category=data.get('category', 'other'),
        description=data.get('description'),
        semester=data['semester'],
        achievement=data.get('achievement'),
        date=date.fromisoformat(data['date']) if data.get('date') else None
    )
    db.session.add(activity)
    db.session.commit()
    return jsonify({'message': 'Activity added', 'activity': activity.to_dict()}), 201


@teacher_bp.route('/activities/<int:student_id>', methods=['GET'])
@jwt_required()
def get_student_activities(student_id):
    user_id = int(get_jwt_identity())
    teacher = get_teacher(user_id)
    if not teacher:
        return jsonify({'error': 'Teacher access required'}), 403

    semester = request.args.get('semester', type=int)
    query = Activity.query.filter_by(student_id=student_id)
    if semester:
        query = query.filter_by(semester=semester)
    return jsonify([a.to_dict() for a in query.all()]), 200
