# 🎓 StudPer — Student Performance Management System

A full-stack web application built with **React + Flask + MySQL + SQLAlchemy**.

---

## 📁 Project Structure

```
student-performance-system/
├── backend/
│   ├── app.py                  ← Flask entry point
│   ├── requirements.txt
│   ├── .env.example
│   ├── models/
│   │   ├── database.py         ← SQLAlchemy db instance
│   │   ├── user.py             ← User model (admin/teacher/student)
│   │   ├── student.py          ← Student profile
│   │   ├── teacher.py          ← Teacher profile
│   │   ├── mark.py             ← Marks model
│   │   └── activity.py         ← Extracurricular activities
│   ├── routes/
│   │   ├── auth.py             ← /api/auth/*
│   │   ├── admin.py            ← /api/admin/*
│   │   ├── teacher.py          ← /api/teacher/*
│   │   └── student.py          ← /api/student/*
│   └── uploads/                ← Student profile photos
│
└── frontend/
    ├── package.json
    ├── public/index.html
    └── src/
        ├── App.jsx             ← Main router
        ├── index.js
        ├── index.css           ← Global styles
        ├── context/
        │   └── AuthContext.jsx ← JWT auth state
        ├── utils/
        │   └── api.js          ← Axios instance
        ├── components/
        │   └── Sidebar.jsx
        └── pages/
            ├── AuthPage.jsx         ← Login + Register
            ├── AdminDashboard.jsx   ← Admin panel
            ├── TeacherDashboard.jsx ← Teacher panel
            └── StudentDashboard.jsx ← Student panel
```

---

## 🛠️ Setup Instructions

### 1. MySQL Database

```sql
CREATE DATABASE student_performance CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/macOS
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
cp .env.example .env
# Edit .env with your MySQL credentials

# Run the server
python app.py
```

Backend runs at: **http://localhost:5000**

On first run, a default admin is auto-created:
- Email: `admin@school.com`
- Password: `admin123`

### 3. Frontend Setup

```bash
cd frontend

npm install
npm start
```

Frontend runs at: **http://localhost:3000**

---

## 🔑 User Roles & Features

### 👑 Admin
- View dashboard stats (students, teachers, pending approvals)
- Approve or reject teacher/student registrations
- View all users by role

### 👨‍🏫 Teacher (requires admin approval)
- View all approved students with semester-wise sorting
- Add/edit/delete marks per subject per student
- Add extracurricular activities with categories and achievements

### 🎓 Student (requires admin approval)
- View and edit personal profile
- Upload/update profile photo
- View semester-wise academic performance with charts:
  - Bar chart (subject-wise scores)
  - Radar chart (performance overview)
  - Trend chart (semester-over-semester)
- View extracurricular activities

---

## 🔌 API Endpoints

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/auth/register | Register student/teacher |
| POST | /api/auth/login | Login |
| GET | /api/auth/me | Get current user |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/admin/pending | List pending users |
| PUT | /api/admin/approve/:id | Approve user |
| DELETE | /api/admin/reject/:id | Reject user |
| GET | /api/admin/users?role=student | List all users |
| GET | /api/admin/stats | Dashboard stats |

### Teacher
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/teacher/students | Get approved students |
| POST | /api/teacher/marks | Add mark |
| GET | /api/teacher/marks/:student_id | Get student marks |
| PUT | /api/teacher/marks/:id | Update mark |
| DELETE | /api/teacher/marks/:id | Delete mark |
| POST | /api/teacher/activities | Add activity |
| GET | /api/teacher/activities/:student_id | Get activities |

### Student
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /api/student/profile | Get profile |
| PUT | /api/student/profile | Update profile |
| POST | /api/student/photo | Upload profile photo |
| GET | /api/student/performance | Get semester-wise data |

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18, React Router v6, Recharts |
| Backend | Python 3.10+, Flask 3, Flask-JWT-Extended |
| Database | MySQL 8 |
| ORM | SQLAlchemy 2, Flask-SQLAlchemy |
| Auth | JWT (JSON Web Tokens) |
| File Upload | Werkzeug, Pillow |
| Charts | Recharts (Bar, Radar, Line) |
| Styling | Custom CSS with CSS Variables |

---

## 🎨 Design

- Dark theme with CSS custom properties
- **Syne** font for headings, **DM Sans** for body
- Responsive layout with sidebar navigation
- Animated performance charts
- Role-based routing and protected pages
