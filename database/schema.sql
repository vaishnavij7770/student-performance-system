-- ============================================================
--  EduTrack — Student Performance Management System
--  Complete MySQL Database Setup Script
--  Run this file once to create all tables and default admin
-- ============================================================

-- 1. Create & select the database
CREATE DATABASE IF NOT EXISTS student_performance
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE student_performance;

-- ============================================================
--  TABLE: users
--  Stores all users: admin, teacher, student
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id            INT            NOT NULL AUTO_INCREMENT,
    name          VARCHAR(100)   NOT NULL,
    email         VARCHAR(150)   NOT NULL,
    password_hash VARCHAR(255)   NOT NULL,
    role          ENUM('admin','teacher','student') NOT NULL,
    is_approved   TINYINT(1)     NOT NULL DEFAULT 0,
    created_at    DATETIME       NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    UNIQUE KEY uq_users_email (email)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
--  TABLE: students
--  Extended profile for users with role = 'student'
-- ============================================================
CREATE TABLE IF NOT EXISTS students (
    id                INT          NOT NULL AUTO_INCREMENT,
    user_id           INT          NOT NULL,
    roll_number       VARCHAR(50)  NOT NULL,
    department        VARCHAR(100)          DEFAULT NULL,
    current_semester  INT          NOT NULL DEFAULT 1,
    photo_url         VARCHAR(255)          DEFAULT NULL,
    date_of_birth     DATE                  DEFAULT NULL,
    phone             VARCHAR(20)           DEFAULT NULL,
    address           TEXT                  DEFAULT NULL,

    PRIMARY KEY (id),
    UNIQUE KEY uq_students_roll (roll_number),
    UNIQUE KEY uq_students_user (user_id),
    CONSTRAINT fk_students_user
        FOREIGN KEY (user_id) REFERENCES users (id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
--  TABLE: teachers
--  Extended profile for users with role = 'teacher'
-- ============================================================
CREATE TABLE IF NOT EXISTS teachers (
    id                      INT          NOT NULL AUTO_INCREMENT,
    user_id                 INT          NOT NULL,
    employee_id             VARCHAR(50)  NOT NULL,
    department              VARCHAR(100)          DEFAULT NULL,
    subject_specialization  VARCHAR(150)          DEFAULT NULL,
    phone                   VARCHAR(20)           DEFAULT NULL,

    PRIMARY KEY (id),
    UNIQUE KEY uq_teachers_emp (employee_id),
    UNIQUE KEY uq_teachers_user (user_id),
    CONSTRAINT fk_teachers_user
        FOREIGN KEY (user_id) REFERENCES users (id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
--  TABLE: marks
--  Academic marks per student, subject, semester
-- ============================================================
CREATE TABLE IF NOT EXISTS marks (
    id              INT     NOT NULL AUTO_INCREMENT,
    student_id      INT     NOT NULL,
    teacher_id      INT     NOT NULL,
    subject         VARCHAR(100)  NOT NULL,
    semester        INT     NOT NULL,
    marks_obtained  FLOAT   NOT NULL,
    total_marks     FLOAT   NOT NULL DEFAULT 100,
    exam_type       ENUM('midterm','final','assignment','quiz') NOT NULL DEFAULT 'final',
    remarks         TEXT             DEFAULT NULL,
    created_at      DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    KEY idx_marks_student   (student_id),
    KEY idx_marks_teacher   (teacher_id),
    KEY idx_marks_semester  (semester),
    CONSTRAINT fk_marks_student
        FOREIGN KEY (student_id) REFERENCES students (id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_marks_teacher
        FOREIGN KEY (teacher_id) REFERENCES teachers (id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
--  TABLE: activities
--  Extracurricular activities per student
-- ============================================================
CREATE TABLE IF NOT EXISTS activities (
    id           INT     NOT NULL AUTO_INCREMENT,
    student_id   INT     NOT NULL,
    teacher_id   INT     NOT NULL,
    title        VARCHAR(200) NOT NULL,
    category     ENUM('sports','cultural','technical','social','academic','other') NOT NULL DEFAULT 'other',
    description  TEXT         DEFAULT NULL,
    semester     INT          NOT NULL,
    achievement  VARCHAR(100) DEFAULT NULL,
    date         DATE         DEFAULT NULL,
    created_at   DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    KEY idx_activities_student  (student_id),
    KEY idx_activities_teacher  (teacher_id),
    KEY idx_activities_semester (semester),
    CONSTRAINT fk_activities_student
        FOREIGN KEY (student_id) REFERENCES students (id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_activities_teacher
        FOREIGN KEY (teacher_id) REFERENCES teachers (id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
--  TABLE: attendance
--  Daily attendance records per student, subject, date
-- ============================================================
CREATE TABLE IF NOT EXISTS attendance (
    id          INT     NOT NULL AUTO_INCREMENT,
    student_id  INT     NOT NULL,
    teacher_id  INT     NOT NULL,
    subject     VARCHAR(100) NOT NULL,
    semester    INT          NOT NULL,
    date        DATE         NOT NULL,
    status      ENUM('present','absent','late') NOT NULL DEFAULT 'present',
    remarks     VARCHAR(200) DEFAULT NULL,
    created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    PRIMARY KEY (id),
    -- Prevent duplicate entry for same student+subject+date
    UNIQUE KEY uq_attendance_record (student_id, subject, date),
    KEY idx_attendance_student  (student_id),
    KEY idx_attendance_teacher  (teacher_id),
    KEY idx_attendance_date     (date),
    KEY idx_attendance_semester (semester),
    CONSTRAINT fk_attendance_student
        FOREIGN KEY (student_id) REFERENCES students (id)
        ON DELETE CASCADE ON UPDATE CASCADE,
    CONSTRAINT fk_attendance_teacher
        FOREIGN KEY (teacher_id) REFERENCES teachers (id)
        ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
--  DEFAULT ADMIN USER
--  Email:    admin@school.com
--  Password: admin123
--  (bcrypt hash of "admin123" — generated by werkzeug)
-- ============================================================
INSERT INTO users (name, email, password_hash, role, is_approved)
SELECT
    'Admin',
    'admin@school.com',
    'scrypt:32768:8:1$placeholder$xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    'admin',
    1
WHERE NOT EXISTS (
    SELECT 1 FROM users WHERE email = 'admin@school.com'
);

-- NOTE: The password hash above is a placeholder.
-- The real hash is auto-generated when you run `python app.py`
-- (the seed_admin() function in app.py handles this correctly).
-- Use `python app.py` to start — it will insert the properly hashed admin.


-- ============================================================
--  VERIFY: show all created tables
-- ============================================================
SHOW TABLES;
