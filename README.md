# Smart Learning Planner 🧠📚

Smart Learning Planner is a full-stack AI-powered web application that helps students
plan their studies effectively, generate AI-based weekly timetables, and track
their learning progress.

---

## 🚀 Features

- User authentication (Signup & Login)
- JWT-based secure authentication
- Smart study plan generator
- AI-powered weekly timetable generation
- Weekly progress tracking
- Protected backend APIs
- Clean and modern UI

---

## 🛠 Tech Stack

### Frontend
- React (Vite)
- Axios
- React Router
- JavaScript
- CSS

### Backend
- FastAPI
- SQLAlchemy
- JWT Authentication
- OpenAI API

### Database
- MySQL

---

## 🔐 Authentication Flow

- User signs up or logs in
- Backend returns JWT token
- Token is stored in browser localStorage
- Token is sent in Authorization header for protected APIs

---

## 📂 Project Structure

smart-learning-planner/
│
├── frontend/
│ ├── src/
│ │ ├── pages/
│ │ ├── components/
│ │ ├── services/
│ │ └── utils/
│
├── backend/
│ ├── app/
│ │ ├── routes/
│ │ ├── models.py
│ │ ├── schemas.py
│ │ ├── database.py
│ │ └── main.py
│
└── README.md


---

## ⚙️ Setup Instructions

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload

Backend will run at:
http://127.0.0.1:8000

Swagger API docs:
http://127.0.0.1:8000/docs

Frontend Setup

cd frontend
npm install
npm run dev

Frontend will run at:
http://localhost:5173

📊 Application Modules
Authentication – Signup & Login

Study Plan – Add and view study plans

AI Timetable – Generate weekly AI timetable

Progress – Track weekly study hours

🧪 Status
✔ Backend working
✔ Frontend working
✔ Authentication secured
✔ AI features integrated

👩‍💻 Author
Vaishnavi Jadhav

GitHub: https://github.com/vaishnavij7770

---

## 📍 STEP 4: SAVE FILE
Press:
CTRL + S

---

## 📍 STEP 5: PUSH README TO GITHUB

Open **Command Prompt** in `smart-learning-planner` folder and run:

```bash
git add README.md
git commit -m "Add professional README"
git push
