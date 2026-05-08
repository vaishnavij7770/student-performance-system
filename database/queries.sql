-- ============================================================
--  EduTrack — Useful Query Reference
--  Handy queries for testing, reporting, and debugging
-- ============================================================

USE student_performance;


-- ============================================================
--  1. GET ALL PENDING USERS (not yet approved)
-- ============================================================
SELECT
    u.id,
    u.name,
    u.email,
    u.role,
    u.created_at,
    COALESCE(s.roll_number, t.employee_id) AS identifier
FROM users u
LEFT JOIN students s ON s.user_id = u.id
LEFT JOIN teachers t ON t.user_id = u.id
WHERE u.is_approved = 0
  AND u.role != 'admin'
ORDER BY u.created_at DESC;


-- ============================================================
--  2. APPROVE A USER (replace ? with the user id)
-- ============================================================
UPDATE users SET is_approved = 1 WHERE id = ?;


-- ============================================================
--  3. GET ALL STUDENTS WITH PROFILE (approved only)
-- ============================================================
SELECT
    u.id          AS user_id,
    u.name,
    u.email,
    s.id          AS student_id,
    s.roll_number,
    s.department,
    s.current_semester,
    s.phone,
    u.created_at
FROM users u
JOIN students s ON s.user_id = u.id
WHERE u.is_approved = 1
ORDER BY s.current_semester, u.name;


-- ============================================================
--  4. GET STUDENTS BY SEMESTER
-- ============================================================
SELECT
    u.name,
    s.roll_number,
    s.department,
    s.current_semester
FROM users u
JOIN students s ON s.user_id = u.id
WHERE u.is_approved = 1
  AND s.current_semester = 2        -- change semester here
ORDER BY u.name;


-- ============================================================
--  5. GET ALL MARKS FOR A STUDENT (replace ? with student id)
-- ============================================================
SELECT
    m.subject,
    m.semester,
    m.marks_obtained,
    m.total_marks,
    ROUND((m.marks_obtained / m.total_marks) * 100, 2) AS percentage,
    m.exam_type,
    m.remarks,
    u.name AS teacher_name,
    m.created_at
FROM marks m
JOIN teachers t ON t.id = m.teacher_id
JOIN users   u ON u.id = t.user_id
WHERE m.student_id = ?
ORDER BY m.semester, m.subject;


-- ============================================================
--  6. SEMESTER-WISE AVERAGE PERCENTAGE FOR A STUDENT
-- ============================================================
SELECT
    m.semester,
    COUNT(m.id)                                          AS total_subjects,
    ROUND(AVG((m.marks_obtained / m.total_marks)*100), 2) AS avg_percentage,
    ROUND(MAX((m.marks_obtained / m.total_marks)*100), 2) AS highest,
    ROUND(MIN((m.marks_obtained / m.total_marks)*100), 2) AS lowest
FROM marks m
WHERE m.student_id = ?          -- replace with student id
GROUP BY m.semester
ORDER BY m.semester;


-- ============================================================
--  7. TOP 10 STUDENTS BY AVERAGE MARKS (all semesters)
-- ============================================================
SELECT
    u.name,
    s.roll_number,
    s.department,
    ROUND(AVG((m.marks_obtained / m.total_marks) * 100), 2) AS overall_avg
FROM marks m
JOIN students s ON s.id = m.student_id
JOIN users   u ON u.id = s.user_id
GROUP BY m.student_id, u.name, s.roll_number, s.department
ORDER BY overall_avg DESC
LIMIT 10;


-- ============================================================
--  8. ATTENDANCE SUMMARY PER STUDENT PER SUBJECT
-- ============================================================
SELECT
    u.name        AS student_name,
    s.roll_number,
    a.subject,
    a.semester,
    SUM(CASE WHEN a.status = 'present' THEN 1 ELSE 0 END) AS present,
    SUM(CASE WHEN a.status = 'absent'  THEN 1 ELSE 0 END) AS absent,
    SUM(CASE WHEN a.status = 'late'    THEN 1 ELSE 0 END) AS late,
    COUNT(*)                                               AS total_classes,
    ROUND(
        (SUM(CASE WHEN a.status IN ('present','late') THEN 1 ELSE 0 END) / COUNT(*)) * 100,
    1)                                                     AS attendance_pct
FROM attendance a
JOIN students s ON s.id = a.student_id
JOIN users    u ON u.id = s.user_id
GROUP BY a.student_id, a.subject, a.semester
ORDER BY u.name, a.semester, a.subject;


-- ============================================================
--  9. STUDENTS WITH ATTENDANCE BELOW 75%
-- ============================================================
SELECT
    u.name        AS student_name,
    s.roll_number,
    a.subject,
    a.semester,
    ROUND(
        (SUM(CASE WHEN a.status IN ('present','late') THEN 1 ELSE 0 END) / COUNT(*)) * 100,
    1) AS attendance_pct
FROM attendance a
JOIN students s ON s.id = a.student_id
JOIN users    u ON u.id = s.user_id
GROUP BY a.student_id, a.subject, a.semester
HAVING attendance_pct < 75
ORDER BY attendance_pct ASC;


-- ============================================================
--  10. ALL ACTIVITIES FOR A STUDENT
-- ============================================================
SELECT
    act.title,
    act.category,
    act.achievement,
    act.semester,
    act.date,
    act.description,
    u.name AS added_by
FROM activities act
JOIN teachers t ON t.id = act.teacher_id
JOIN users    u ON u.id = t.user_id
WHERE act.student_id = ?          -- replace with student id
ORDER BY act.semester, act.date DESC;


-- ============================================================
--  11. ADMIN DASHBOARD STATS
-- ============================================================
SELECT
    (SELECT COUNT(*) FROM users WHERE role='student')                         AS total_students,
    (SELECT COUNT(*) FROM users WHERE role='student' AND is_approved=1)       AS approved_students,
    (SELECT COUNT(*) FROM users WHERE role='teacher')                         AS total_teachers,
    (SELECT COUNT(*) FROM users WHERE role='teacher' AND is_approved=1)       AS approved_teachers,
    (SELECT COUNT(*) FROM users WHERE role!='admin' AND is_approved=0)        AS pending_approvals,
    (SELECT COUNT(*) FROM marks)                                               AS total_marks_entries,
    (SELECT COUNT(*) FROM activities)                                          AS total_activities,
    (SELECT COUNT(*) FROM attendance)                                          AS total_attendance_records;


-- ============================================================
--  12. SUBJECT-WISE PERFORMANCE ACROSS ALL STUDENTS
-- ============================================================
SELECT
    m.subject,
    m.semester,
    COUNT(DISTINCT m.student_id)                                      AS students_appeared,
    ROUND(AVG((m.marks_obtained / m.total_marks) * 100), 2)           AS avg_pct,
    ROUND(MAX((m.marks_obtained / m.total_marks) * 100), 2)           AS highest_pct,
    ROUND(MIN((m.marks_obtained / m.total_marks) * 100), 2)           AS lowest_pct,
    SUM(CASE WHEN (m.marks_obtained/m.total_marks)*100 >= 75 THEN 1 ELSE 0 END) AS passed,
    SUM(CASE WHEN (m.marks_obtained/m.total_marks)*100 <  40 THEN 1 ELSE 0 END) AS failed
FROM marks m
GROUP BY m.subject, m.semester
ORDER BY m.semester, m.subject;


-- ============================================================
--  13. RESET / DROP ALL TABLES (USE WITH CAUTION)
-- ============================================================
/*
SET FOREIGN_KEY_CHECKS = 0;
DROP TABLE IF EXISTS attendance;
DROP TABLE IF EXISTS activities;
DROP TABLE IF EXISTS marks;
DROP TABLE IF EXISTS teachers;
DROP TABLE IF EXISTS students;
DROP TABLE IF EXISTS users;
SET FOREIGN_KEY_CHECKS = 1;
*/
