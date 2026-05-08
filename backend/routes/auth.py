from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from models.database import db
from models.user import User
from models.student import Student
from models.teacher import Teacher

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    required = ['name', 'email', 'password', 'role']
    for field in required:
        if not data.get(field):
            return jsonify({'error': f'{field} is required'}), 400

    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 409

    user = User(name=data['name'], email=data['email'], role=data['role'])
    user.set_password(data['password'])

    # Admin auto-approved
    if data['role'] == 'admin':
        user.is_approved = True

    db.session.add(user)
    db.session.flush()

    if data['role'] == 'student':
        if not data.get('roll_number'):
            return jsonify({'error': 'roll_number is required for students'}), 400
        student = Student(
            user_id=user.id,
            roll_number=data['roll_number'],
            department=data.get('department', ''),
            current_semester=data.get('current_semester', 1),
            phone=data.get('phone'),
            date_of_birth=data.get('date_of_birth'),
            address=data.get('address')
        )
        db.session.add(student)

    elif data['role'] == 'teacher':
        if not data.get('employee_id'):
            return jsonify({'error': 'employee_id is required for teachers'}), 400
        teacher = Teacher(
            user_id=user.id,
            employee_id=data['employee_id'],
            department=data.get('department', ''),
            subject_specialization=data.get('subject_specialization', ''),
            phone=data.get('phone')
        )
        db.session.add(teacher)

    db.session.commit()
    return jsonify({'message': 'Registration successful. Awaiting admin approval.', 'user': user.to_dict()}), 201


@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    if not data or not data.get('email') or not data.get('password'):
        return jsonify({'error': 'Email and password are required'}), 400

    user = User.query.filter_by(email=data['email'].strip().lower()).first()

    # Also try original case if lowercase fails
    if not user:
        user = User.query.filter_by(email=data['email'].strip()).first()

    if not user:
        return jsonify({'error': 'No account found with this email'}), 401

    if not user.check_password(data['password']):
        return jsonify({'error': 'Incorrect password'}), 401

    if not user.is_approved and user.role != 'admin':
        return jsonify({'error': 'Account pending admin approval. Please ask admin to approve your account.'}), 403

    token = create_access_token(identity=str(user.id))
    profile = None
    if user.role == 'student' and user.student_profile:
        profile = user.student_profile.to_dict()
    elif user.role == 'teacher' and user.teacher_profile:
        profile = user.teacher_profile.to_dict()

    return jsonify({'token': token, 'user': user.to_dict(), 'profile': profile}), 200


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    user_id = int(get_jwt_identity())
    user = User.query.get_or_404(user_id)
    profile = None
    if user.role == 'student' and user.student_profile:
        profile = user.student_profile.to_dict()
    elif user.role == 'teacher' and user.teacher_profile:
        profile = user.teacher_profile.to_dict()
    return jsonify({'user': user.to_dict(), 'profile': profile}), 200


# ── Development-only debug route ─────────────────────────────
# Visit: http://localhost:5000/api/auth/debug-users
# Shows all users and their approval status (REMOVE IN PRODUCTION)
@auth_bp.route('/debug-users', methods=['GET'])
def debug_users():
    users = User.query.all()
    return jsonify([{
        'id': u.id,
        'name': u.name,
        'email': u.email,
        'role': u.role,
        'is_approved': u.is_approved,
        'hash_prefix': u.password_hash[:30] + '...' if u.password_hash else None
    } for u in users]), 200


# ── Fix: force-reset admin password ──────────────────────────
# Visit: http://localhost:5000/api/auth/fix-admin
# Sets admin@school.com password back to admin123
@auth_bp.route('/fix-admin', methods=['GET'])
def fix_admin():
    admin = User.query.filter_by(email='admin@school.com').first()
    if not admin:
        # Create admin if missing
        admin = User(name='Admin', email='admin@school.com', role='admin', is_approved=True)
        db.session.add(admin)
    admin.set_password('admin123')
    admin.is_approved = True
    db.session.commit()
    return jsonify({'message': 'Admin password reset to admin123 ✅', 'email': admin.email}), 200
