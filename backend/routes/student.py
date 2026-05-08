from flask import Blueprint, request, jsonify, current_app, send_from_directory
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.database import db
from models.user import User
from models.student import Student
from models.mark import Mark
from models.activity import Activity
from werkzeug.utils import secure_filename
import os

student_bp = Blueprint('student', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@student_bp.route('/profile', methods=['GET'])
@jwt_required()
def get_profile():
    user_id = int(get_jwt_identity())
    user = User.query.get_or_404(user_id)
    if user.role != 'student':
        return jsonify({'error': 'Student access required'}), 403
    return jsonify({'user': user.to_dict(), 'profile': user.student_profile.to_dict()}), 200


@student_bp.route('/profile', methods=['PUT'])
@jwt_required()
def update_profile():
    user_id = int(get_jwt_identity())
    user = User.query.get_or_404(user_id)
    if user.role != 'student':
        return jsonify({'error': 'Student access required'}), 403

    data = request.get_json()
    student = user.student_profile

    for field in ['phone', 'address', 'department']:
        if field in data:
            setattr(student, field, data[field])

    if 'name' in data:
        user.name = data['name']

    db.session.commit()
    return jsonify({'message': 'Profile updated', 'profile': student.to_dict()}), 200


@student_bp.route('/photo', methods=['POST'])
@jwt_required()
def upload_photo():
    user_id = int(get_jwt_identity())
    user = User.query.get_or_404(user_id)
    if user.role != 'student':
        return jsonify({'error': 'Student access required'}), 403

    if 'photo' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['photo']
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(f"student_{user_id}_{file.filename}")
    upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)

    user.student_profile.photo_url = f"/uploads/{filename}"
    db.session.commit()
    return jsonify({'message': 'Photo uploaded', 'photo_url': user.student_profile.photo_url}), 200


@student_bp.route('/performance', methods=['GET'])
@jwt_required()
def get_performance():
    user_id = int(get_jwt_identity())
    user = User.query.get_or_404(user_id)
    if user.role != 'student':
        return jsonify({'error': 'Student access required'}), 403

    student = user.student_profile
    semester = request.args.get('semester', type=int)

    marks_query = Mark.query.filter_by(student_id=student.id)
    activities_query = Activity.query.filter_by(student_id=student.id)

    if semester:
        marks_query = marks_query.filter_by(semester=semester)
        activities_query = activities_query.filter_by(semester=semester)

    marks = marks_query.all()
    activities = activities_query.all()

    # Group marks by semester
    semester_data = {}
    for m in marks:
        sem = str(m.semester)
        if sem not in semester_data:
            semester_data[sem] = {'marks': [], 'activities': []}
        semester_data[sem]['marks'].append(m.to_dict())

    for a in activities:
        sem = str(a.semester)
        if sem not in semester_data:
            semester_data[sem] = {'marks': [], 'activities': []}
        semester_data[sem]['activities'].append(a.to_dict())

    return jsonify({'semester_data': semester_data, 'student': student.to_dict()}), 200


@student_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
