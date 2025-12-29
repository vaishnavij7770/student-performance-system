import streamlit as st 
import pandas as pd
import numpy as np
import os
import hashlib
import pickle
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# =========================
# DATABASE (MySQL ONLY)
# =========================

DB_USER = "root"
DB_PASSWORD = "12345678"
DB_HOST = "localhost"
DB_NAME = "student_performance"

engine = create_engine(
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
    echo=False
)

Base = declarative_base()
Session = sessionmaker(bind=engine)
db = Session()


# =========================
# TABLE MODELS 
# =========================

class Student(Base):
    __tablename__ = "students"
    username = Column(String(100), primary_key=True)
    password = Column(String(255))
    college_name = Column(String(255))
    class_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)

class Faculty(Base):
    __tablename__ = "faculty"
    username = Column(String(100), primary_key=True)
    password = Column(String(255))
    college_name = Column(String(255))
    department = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)

# CREATE TABLES IF NOT EXISTS
Base.metadata.create_all(engine)


# =========================
# AUTHENTICATION 
# =========================

class UserAuthentication:

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def signup(self, username, password, role, college_name=None, class_or_dept=None):
        hashed = self.hash_password(password)

        if role == "student":
            if db.query(Student).filter_by(username=username).first():
                return False, "Student already exists"

            user = Student(
                username=username,
                password=hashed,
                college_name=college_name,
                class_name=class_or_dept,
                created_at=datetime.now()
            )

        else:  # faculty
            if db.query(Faculty).filter_by(username=username).first():
                return False, "Faculty already exists"

            user = Faculty(
                username=username,
                password=hashed,
                college_name=college_name,
                department=class_or_dept,
                created_at=datetime.now()
            )

        db.add(user)
        db.commit()
        return True, "Account created successfully"

    def login(self, username, password, role):
        hashed = self.hash_password(password)

        if role == "student":
            user = db.query(Student).filter_by(username=username).first()
        else:
            user = db.query(Faculty).filter_by(username=username).first()

        if not user:
            return False, "User not found"
        if user.password != hashed:
            return False, "Invalid password"

        return True, "Login successful"


# =========================
# MACHINE LEARNING
# =========================

class MLPerformancePredictor:
    def __init__(self):
        self.features = [
            'study_hours', 'attendance', 'previous_score',
            'extracurricular', 'sleep_hours',
            'family_support', 'internet_access'
        ]

        self.model_file = "student_model.pkl"
        self.encoder_file = "label_encoders.pkl"
        self.model = None
        self.encoders = {}

    def generate_data(self, n=1000):
        np.random.seed(42)
        df = pd.DataFrame({
            "study_hours": np.random.randint(1, 11, n),
            "attendance": np.random.randint(50, 101, n),
            "previous_score": np.random.randint(40, 101, n),
            "extracurricular": np.random.choice(["Yes", "No"], n),
            "sleep_hours": np.random.randint(4, 11, n),
            "family_support": np.random.choice(["High", "Medium", "Low"], n),
            "internet_access": np.random.choice(["Yes", "No"], n)
        })

        score = (
            df.study_hours * 3 +
            df.attendance * 0.3 +
            df.previous_score * 0.4 +
            df.sleep_hours * 2
        )

        df["performance"] = np.where(
            score > 120, "Excellent",
            np.where(score > 90, "Good",
                     np.where(score > 60, "Average", "Poor"))
        )

        return df

    def train_model(self):
        df = self.generate_data()
        X = df[self.features].copy()
        y = df["performance"]

        for col in X.select_dtypes(include="object"):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.encoders[col] = le

        y_enc = LabelEncoder()
        y = y_enc.fit_transform(y)
        self.encoders["performance"] = y_enc

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

        acc = accuracy_score(y_test, self.model.predict(X_test))

        with open(self.model_file, "wb") as f:
            pickle.dump(self.model, f)

        with open(self.encoder_file, "wb") as f:
            pickle.dump(self.encoders, f)

        return acc

    def load_model(self):
        if os.path.exists(self.model_file) and os.path.exists(self.encoder_file):
            with open(self.model_file, "rb") as f:
                self.model = pickle.load(f)
            with open(self.encoder_file, "rb") as f:
                self.encoders = pickle.load(f)
            return True
        return False

    def predict(self, data):
        if self.model is None:
            if not self.load_model():
                self.train_model()

        for k in self.encoders:
            if k in data:
                data[k] = self.encoders[k].transform([data[k]])[0]

        X = pd.DataFrame([data])[self.features]
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]

        return self.encoders["performance"].inverse_transform([pred])[0], max(prob)*100



# =========================
# STREAMLIT UI
# =========================

st.set_page_config("Student Performance System")
st.title("🎓 Student Performance Analysis ")

auth = UserAuthentication()
predictor = MLPerformancePredictor()

if "logged" not in st.session_state:
    st.session_state.logged = False
    st.session_state.role = None


menu = ["Home", "Student Signup", "Faculty Signup", "Student Login", "Faculty Login"]
if st.session_state.logged:
    menu = ["Dashboard", "Logout"]

choice = st.sidebar.selectbox("Menu", menu)


# ================= HOME =================
if choice == "Home":
    st.info("Welcome to Student Performance Prediction System.")


# ================= STUDENT SIGNUP (UPDATED) =================
elif choice == "Student Signup":
    u = st.text_input("Username")
    college = st.text_input("College Name")
    student_class = st.text_input("Class")
    p = st.text_input("Password", type="password")
    cp = st.text_input("Confirm Password", type="password")

    if st.button("Signup"):
        if p != cp:
            st.error("Passwords do not match!")
        else:
            ok, msg = auth.signup(u, p, "student", college, student_class)
            if ok:
                st.success(msg)
            else:
                st.error(msg)


# ================= FACULTY SIGNUP (UPDATED) =================
elif choice == "Faculty Signup":
    u = st.text_input("Username")
    college = st.text_input("College Name")
    department = st.text_input("Department")
    p = st.text_input("Password", type="password")
    cp = st.text_input("Confirm Password", type="password")

    if st.button("Signup"):
        if p != cp:
            st.error("Passwords do not match!")
        else:
            ok, msg = auth.signup(u, p, "faculty", college, department)
            if ok:
                st.success(msg)
            else:
                st.error(msg)


# ================= LOGIN =================
elif choice == "Student Login":
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        ok, msg = auth.login(u, p, "student")
        if ok:
            st.session_state.logged = True
            st.session_state.role = "student"
            st.success(msg)
        else:
            st.error(msg)

elif choice == "Faculty Login":
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        ok, msg = auth.login(u, p, "faculty")
        if ok:
            st.session_state.logged = True
            st.session_state.role = "faculty"
            st.success(msg)
        else:
            st.error(msg)


# ================= DASHBOARD =================
elif choice == "Dashboard":
    st.subheader("📊 Enter Student Data for Performance Prediction")

    data = {
        "study_hours": st.number_input("Study Hours", 1, 10, 5),
        "attendance": st.number_input("Attendance %", 0, 100, 75),
        "previous_score": st.number_input("Previous Score", 0, 100, 70),
        "sleep_hours": st.number_input("Sleep Hours", 4, 10, 7),
        "extracurricular": st.selectbox("Extracurricular", ["Yes", "No"]),
        "family_support": st.selectbox("Family Support", ["High", "Medium", "Low"]),
        "internet_access": st.selectbox("Internet Access", ["Yes", "No"])
    }

    if st.button("Predict"):
        result, confidence = predictor.predict(data)
        st.success(f"Predicted Performance: {result}")
        st.info(f"Confidence Level: {confidence:.2f}%")


# ================= LOGOUT =================
elif choice == "Logout":
    st.session_state.logged = False
    st.session_state.role = None
    st.success("Logged out successfully!")
