"""
Run this script to populate the database with demo data.
Usage: python seed_demo.py
"""
from app import create_app
from models.database import db
from models.user import User
from models.student import Student
from models.teacher import Teacher
from models.mark import Mark
from models.activity import Activity
from datetime import date, timedelta
import random

app = create_app()

SUBJECTS = {
    1: ['Mathematics I', 'Physics', 'English', 'Programming Basics'],
    2: ['Mathematics II', 'Chemistry', 'Data Structures', 'Digital Electronics'],
    3: ['Algorithms', 'Database Systems', 'Operating Systems', 'Statistics'],
    4: ['Computer Networks', 'Software Engineering', 'Web Development', 'AI Basics'],
}

ACTIVITIES = [
    ('District Chess Championship', 'sports', '1st Place'),
    ('Cultural Fest Dance', 'cultural', 'Best Performance'),
    ('Hackathon 2024', 'technical', '2nd Runner Up'),
    ('NSS Social Service Camp', 'social', 'Participated'),
    ('Coding Olympiad', 'academic', 'Silver Medal'),
    ('Cricket Tournament', 'sports', 'Team Captain'),
    ('Debate Competition', 'academic', 'Best Speaker'),
    ('Art Exhibition', 'cultural', 'Participated'),
]


def seed():
    with app.app_context():
        print("🌱 Seeding demo data...")

        # Create 2 teachers
        teachers = []
        for i in range(1, 3):
            email = f"teacher{i}@school.com"
            if not User.query.filter_by(email=email).first():
                u = User(name=f"Prof. Teacher {i}", email=email, role='teacher', is_approved=True)
                u.set_password('teacher123')
                db.session.add(u)
                db.session.flush()
                t = Teacher(user_id=u.id, employee_id=f"TCH00{i}", department="Computer Science", subject_specialization="Mathematics, Programming")
                db.session.add(t)
                db.session.flush()
                teachers.append(t)
                print(f"  ✅ Created teacher: {email} / teacher123")
            else:
                u = User.query.filter_by(email=email).first()
                if u.teacher_profile:
                    teachers.append(u.teacher_profile)

        db.session.commit()

        # Create 10 students across semesters 1-4
        students = []
        for i in range(1, 11):
            email = f"student{i}@school.com"
            sem = ((i - 1) % 4) + 1
            if not User.query.filter_by(email=email).first():
                u = User(name=f"Student {i}", email=email, role='student', is_approved=True)
                u.set_password('student123')
                db.session.add(u)
                db.session.flush()
                s = Student(
                    user_id=u.id,
                    roll_number=f"CS2024{i:03d}",
                    department="Computer Science",
                    current_semester=sem,
                    phone=f"+91 9000000{i:03d}"
                )
                db.session.add(s)
                db.session.flush()
                students.append(s)
                print(f"  ✅ Created student: {email} / student123 (Sem {sem})")
            else:
                u = User.query.filter_by(email=email).first()
                if u.student_profile:
                    students.append(u.student_profile)

        db.session.commit()

        # Add marks if teachers and students exist
        if teachers and students:
            teacher = teachers[0]
            for student in students:
                sem = student.current_semester
                subj_list = SUBJECTS.get(sem, SUBJECTS[1])
                for subj in subj_list:
                    if not Mark.query.filter_by(student_id=student.id, subject=subj, semester=sem).first():
                        pct = random.uniform(45, 98)
                        mark = Mark(
                            student_id=student.id,
                            teacher_id=teacher.id,
                            subject=subj,
                            semester=sem,
                            marks_obtained=round(pct, 1),
                            total_marks=100,
                            exam_type=random.choice(['final', 'midterm', 'assignment']),
                            remarks='Good performance' if pct >= 75 else 'Needs improvement' if pct < 60 else None
                        )
                        db.session.add(mark)

            # Add marks for previous semesters too
            for student in students[:5]:
                for prev_sem in range(1, student.current_semester):
                    subj_list = SUBJECTS.get(prev_sem, SUBJECTS[1])
                    for subj in subj_list:
                        if not Mark.query.filter_by(student_id=student.id, subject=subj, semester=prev_sem).first():
                            pct = random.uniform(50, 95)
                            mark = Mark(
                                student_id=student.id,
                                teacher_id=teacher.id,
                                subject=subj,
                                semester=prev_sem,
                                marks_obtained=round(pct, 1),
                                total_marks=100,
                                exam_type='final'
                            )
                            db.session.add(mark)

            db.session.commit()
            print(f"  ✅ Added marks for all students")

            # Add activities
            for student in students[:6]:
                acts = random.sample(ACTIVITIES, k=random.randint(2, 4))
                for title, category, achievement in acts:
                    if not Activity.query.filter_by(student_id=student.id, title=title).first():
                        act = Activity(
                            student_id=student.id,
                            teacher_id=teacher.id,
                            title=title,
                            category=category,
                            semester=student.current_semester,
                            achievement=achievement,
                            date=date.today() - timedelta(days=random.randint(10, 180))
                        )
                        db.session.add(act)

            db.session.commit()
            print(f"  ✅ Added activities for students")

        print("\n🎉 Seeding complete!")
        print("\nDemo credentials:")
        print("  Admin:   admin@school.com   / admin123")
        print("  Teacher: teacher1@school.com / teacher123")
        print("  Student: student1@school.com / student123")
        print("           student2@school.com / student123  (etc. up to student10)")


if __name__ == '__main__':
    seed()
