-- ============================================================
--  EduTrack — Demo Data Inserts
--  Run AFTER schema.sql
--  Passwords are all "teacher123" or "student123"
--  (These hashes are placeholders — use seed_demo.py instead
--   for correctly hashed passwords via Flask/Werkzeug)
-- ============================================================

USE student_performance;

-- ============================================================
--  DEMO TEACHERS  (2 teachers)
-- ============================================================
INSERT INTO users (name, email, password_hash, role, is_approved) VALUES
('Prof. Ramesh Kumar',   'teacher1@school.com', 'HASH_PLACEHOLDER', 'teacher', 1),
('Prof. Sunita Sharma',  'teacher2@school.com', 'HASH_PLACEHOLDER', 'teacher', 1);

INSERT INTO teachers (user_id, employee_id, department, subject_specialization, phone)
SELECT u.id, CONCAT('TCH00', ROW_NUMBER() OVER (ORDER BY u.id)),
       'Computer Science', 'Mathematics, Algorithms', '+91 9000000001'
FROM users u WHERE u.role = 'teacher' AND u.email IN ('teacher1@school.com','teacher2@school.com');


-- ============================================================
--  DEMO STUDENTS  (10 students, spread across sem 1–4)
-- ============================================================
INSERT INTO users (name, email, password_hash, role, is_approved) VALUES
('Aarav Mehta',    'student1@school.com',  'HASH_PLACEHOLDER', 'student', 1),
('Priya Nair',     'student2@school.com',  'HASH_PLACEHOLDER', 'student', 1),
('Rohit Verma',    'student3@school.com',  'HASH_PLACEHOLDER', 'student', 1),
('Sneha Patel',    'student4@school.com',  'HASH_PLACEHOLDER', 'student', 1),
('Karan Singh',    'student5@school.com',  'HASH_PLACEHOLDER', 'student', 1),
('Anjali Rao',     'student6@school.com',  'HASH_PLACEHOLDER', 'student', 1),
('Vikram Joshi',   'student7@school.com',  'HASH_PLACEHOLDER', 'student', 1),
('Divya Iyer',     'student8@school.com',  'HASH_PLACEHOLDER', 'student', 1),
('Arjun Gupta',    'student9@school.com',  'HASH_PLACEHOLDER', 'student', 1),
('Meera Pillai',   'student10@school.com', 'HASH_PLACEHOLDER', 'student', 1);

INSERT INTO students (user_id, roll_number, department, current_semester, phone)
VALUES
((SELECT id FROM users WHERE email='student1@school.com'),  'CS2024001', 'Computer Science', 1, '+91 9100000001'),
((SELECT id FROM users WHERE email='student2@school.com'),  'CS2024002', 'Computer Science', 2, '+91 9100000002'),
((SELECT id FROM users WHERE email='student3@school.com'),  'CS2024003', 'Computer Science', 3, '+91 9100000003'),
((SELECT id FROM users WHERE email='student4@school.com'),  'CS2024004', 'Computer Science', 4, '+91 9100000004'),
((SELECT id FROM users WHERE email='student5@school.com'),  'CS2024005', 'Computer Science', 1, '+91 9100000005'),
((SELECT id FROM users WHERE email='student6@school.com'),  'CS2024006', 'Computer Science', 2, '+91 9100000006'),
((SELECT id FROM users WHERE email='student7@school.com'),  'CS2024007', 'Computer Science', 3, '+91 9100000007'),
((SELECT id FROM users WHERE email='student8@school.com'),  'CS2024008', 'Computer Science', 4, '+91 9100000008'),
((SELECT id FROM users WHERE email='student9@school.com'),  'CS2024009', 'Computer Science', 1, '+91 9100000009'),
((SELECT id FROM users WHERE email='student10@school.com'), 'CS2024010', 'Computer Science', 2, '+91 9100000010');


-- ============================================================
--  DEMO MARKS  (semester 1 subjects for students 1 & 5)
-- ============================================================
INSERT INTO marks (student_id, teacher_id, subject, semester, marks_obtained, total_marks, exam_type, remarks)
VALUES
-- Student 1 (Aarav Mehta) - Semester 1
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Mathematics I',    1, 88, 100, 'final',      'Excellent work'),
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Physics',          1, 74, 100, 'final',      'Good effort'),
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'English',          1, 91, 100, 'final',      NULL),
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Programming Basics',1,95, 100, 'final',      'Outstanding'),

-- Student 2 (Priya Nair) - Semester 2
((SELECT id FROM students WHERE roll_number='CS2024002'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Mathematics II',   2, 79, 100, 'final',      NULL),
((SELECT id FROM students WHERE roll_number='CS2024002'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Data Structures',  2, 85, 100, 'midterm',    'Very good'),
((SELECT id FROM students WHERE roll_number='CS2024002'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Chemistry',        2, 61, 100, 'final',      'Needs improvement'),
((SELECT id FROM students WHERE roll_number='CS2024002'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Digital Electronics',2,70,100, 'final',      NULL),

-- Student 3 (Rohit Verma) - Semester 3
((SELECT id FROM students WHERE roll_number='CS2024003'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Algorithms',       3, 55, 100, 'final',      'Needs improvement'),
((SELECT id FROM students WHERE roll_number='CS2024003'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Database Systems', 3, 82, 100, 'final',      NULL),
((SELECT id FROM students WHERE roll_number='CS2024003'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Operating Systems',3, 78, 100, 'midterm',    NULL),
((SELECT id FROM students WHERE roll_number='CS2024003'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Statistics',       3, 90, 100, 'final',      'Excellent'),

-- Student 4 (Sneha Patel) - Semester 4
((SELECT id FROM students WHERE roll_number='CS2024004'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Computer Networks',4, 93, 100, 'final',      'Outstanding'),
((SELECT id FROM students WHERE roll_number='CS2024004'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Software Engineering',4,87,100,'final',      NULL),
((SELECT id FROM students WHERE roll_number='CS2024004'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Web Development',  4, 96, 100, 'assignment', 'Best in class'),
((SELECT id FROM students WHERE roll_number='CS2024004'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'AI Basics',        4, 72, 100, 'final',      NULL);


-- ============================================================
--  DEMO ACTIVITIES
-- ============================================================
INSERT INTO activities (student_id, teacher_id, title, category, description, semester, achievement, date)
VALUES
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'District Chess Championship', 'sports',
 'Represented college at district level chess tournament',
 1, '1st Place', '2024-09-15'),

((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Hackathon 2024', 'technical',
 '24-hour college hackathon — built a smart attendance system',
 1, '2nd Runner Up', '2024-10-20'),

((SELECT id FROM students WHERE roll_number='CS2024002'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Cultural Fest Dance', 'cultural',
 'Classical dance performance at annual cultural festival',
 2, 'Best Performance', '2024-11-05'),

((SELECT id FROM students WHERE roll_number='CS2024003'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'NSS Social Service Camp', 'social',
 'Participated in 7-day NSS camp for rural community service',
 3, 'Participated', '2024-08-10'),

((SELECT id FROM students WHERE roll_number='CS2024004'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Coding Olympiad', 'academic',
 'State level competitive programming contest',
 4, 'Silver Medal', '2024-12-01'),

((SELECT id FROM students WHERE roll_number='CS2024004'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Cricket Tournament', 'sports',
 'Inter-college cricket tournament',
 4, 'Team Captain', '2024-09-22');


-- ============================================================
--  DEMO ATTENDANCE  (3 days for student 1)
-- ============================================================
INSERT INTO attendance (student_id, teacher_id, subject, semester, date, status, remarks)
VALUES
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Mathematics I', 1, '2024-09-02', 'present', NULL),
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Mathematics I', 1, '2024-09-04', 'present', NULL),
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Mathematics I', 1, '2024-09-06', 'absent',  'Sick leave'),
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Mathematics I', 1, '2024-09-09', 'present', NULL),
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Programming Basics', 1, '2024-09-03', 'present', NULL),
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Programming Basics', 1, '2024-09-05', 'late',    'Arrived 10 min late'),
((SELECT id FROM students WHERE roll_number='CS2024001'),
 (SELECT id FROM teachers WHERE employee_id='TCH001'),
 'Programming Basics', 1, '2024-09-10', 'present', NULL);


-- ============================================================
--  FINAL CHECK
-- ============================================================
SELECT 'users'      AS tbl, COUNT(*) AS rows FROM users
UNION ALL
SELECT 'students',          COUNT(*) FROM students
UNION ALL
SELECT 'teachers',          COUNT(*) FROM teachers
UNION ALL
SELECT 'marks',             COUNT(*) FROM marks
UNION ALL
SELECT 'activities',        COUNT(*) FROM activities
UNION ALL
SELECT 'attendance',        COUNT(*) FROM attendance;
