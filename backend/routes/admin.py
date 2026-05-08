from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.database import db
from models.user import User
from models.student import Student
from models.teacher import Teacher

admin_bp = Blueprint('admin', __name__)

def require_admin():
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    if not user or user.role != 'admin':
        return None, jsonify({'error': 'Admin access required'}), 403
    return user, None, None


@admin_bp.route('/pending', methods=['GET'])
@jwt_required()
def get_pending():
    user_id = int(get_jwt_identity())
    admin = User.query.get(user_id)
    if not admin or admin.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403

    pending = User.query.filter_by(is_approved=False).filter(User.role != 'admin').all()
    result = []
    for u in pending:
        d = u.to_dict()
        if u.role == 'student' and u.student_profile:
            d['profile'] = u.student_profile.to_dict()
        elif u.role == 'teacher' and u.teacher_profile:
            d['profile'] = u.teacher_profile.to_dict()
        result.append(d)
    return jsonify(result), 200


@admin_bp.route('/approve/<int:user_id>', methods=['PUT'])
@jwt_required()
def approve_user(user_id):
    requester_id = int(get_jwt_identity())
    admin = User.query.get(requester_id)
    if not admin or admin.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403

    user = User.query.get_or_404(user_id)
    user.is_approved = True
    db.session.commit()
    return jsonify({'message': f'{user.name} approved successfully', 'user': user.to_dict()}), 200


@admin_bp.route('/reject/<int:user_id>', methods=['DELETE'])
@jwt_required()
def reject_user(user_id):
    requester_id = int(get_jwt_identity())
    admin = User.query.get(requester_id)
    if not admin or admin.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403

    user = User.query.get_or_404(user_id)
    if user.student_profile:
        db.session.delete(user.student_profile)
    if user.teacher_profile:
        db.session.delete(user.teacher_profile)
    db.session.delete(user)
    db.session.commit()
    return jsonify({'message': 'User rejected and removed'}), 200


@admin_bp.route('/users', methods=['GET'])
@jwt_required()
def get_all_users():
    requester_id = int(get_jwt_identity())
    admin = User.query.get(requester_id)
    if not admin or admin.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403

    role = request.args.get('role')
    query = User.query
    if role:
        query = query.filter_by(role=role)
    users = query.all()
    result = []
    for u in users:
        d = u.to_dict()
        if u.role == 'student' and u.student_profile:
            d['profile'] = u.student_profile.to_dict()
        elif u.role == 'teacher' and u.teacher_profile:
            d['profile'] = u.teacher_profile.to_dict()
        result.append(d)
    return jsonify(result), 200


@admin_bp.route('/stats', methods=['GET'])
@jwt_required()
def get_stats():
    requester_id = int(get_jwt_identity())
    admin = User.query.get(requester_id)
    if not admin or admin.role != 'admin':
        return jsonify({'error': 'Admin access required'}), 403

    return jsonify({
        'total_students': User.query.filter_by(role='student').count(),
        'approved_students': User.query.filter_by(role='student', is_approved=True).count(),
        'total_teachers': User.query.filter_by(role='teacher').count(),
        'approved_teachers': User.query.filter_by(role='teacher', is_approved=True).count(),
        'pending_approvals': User.query.filter_by(is_approved=False).filter(User.role != 'admin').count(),
    }), 200
