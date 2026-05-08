import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import API from '../utils/api';

export default function AuthPage() {
  const [tab, setTab] = useState('login');
  const [role, setRole] = useState('student');
  const [form, setForm] = useState({});
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const set = (k, v) => setForm(f => ({ ...f, [k]: v }));

  const handleLogin = async (e) => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      const data = await login(form.email, form.password);
      if (data.user.role === 'admin') navigate('/admin');
      else if (data.user.role === 'teacher') navigate('/teacher');
      else navigate('/student');
    } catch (err) {
      setError(err.response?.data?.error || 'Login failed');
    } finally { setLoading(false); }
  };

  const handleRegister = async (e) => {
    e.preventDefault();
    setError(''); setLoading(true);
    try {
      await API.post('/auth/register', { ...form, role });
      setSuccess('Registration successful! Awaiting admin approval.');
      setForm({});
      setTab('login');
    } catch (err) {
      setError(err.response?.data?.error || 'Registration failed');
    } finally { setLoading(false); }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">
          <h1>🎓 StudPer</h1>
          <p>Student Performance System</p>
        </div>

        <div className="auth-tabs">
          <button className={`auth-tab ${tab === 'login' ? 'active' : ''}`} onClick={() => { setTab('login'); setError(''); setSuccess(''); }}>Sign In</button>
          <button className={`auth-tab ${tab === 'register' ? 'active' : ''}`} onClick={() => { setTab('register'); setError(''); setSuccess(''); }}>Register</button>
        </div>

        {error && <div className="error-msg">⚠️ {error}</div>}
        {success && <div className="success-msg">✅ {success}</div>}

        {tab === 'login' ? (
          <form onSubmit={handleLogin}>
            <div className="form-group">
              <label className="form-label">Email</label>
              <input className="form-input" type="email" placeholder="your@email.com" value={form.email || ''} onChange={e => set('email', e.target.value)} required />
            </div>
            <div className="form-group">
              <label className="form-label">Password</label>
              <input className="form-input" type="password" placeholder="••••••••" value={form.password || ''} onChange={e => set('password', e.target.value)} required />
            </div>
            <button className="btn btn-primary w-full" type="submit" disabled={loading} style={{ marginTop: 8, justifyContent: 'center' }}>
              {loading ? <span className="loading-spinner" /> : '→ Sign In'}
            </button>
            <p style={{ textAlign: 'center', marginTop: 16, fontSize: '0.78rem', color: 'var(--text3)' }}>
              Default admin: admin@school.com / admin123
            </p>
          </form>
        ) : (
          <form onSubmit={handleRegister}>
            <div className="form-group">
              <label className="form-label">Register As</label>
              <select className="form-select" value={role} onChange={e => setRole(e.target.value)}>
                <option value="student">Student</option>
                <option value="teacher">Teacher</option>
              </select>
            </div>
            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">Full Name</label>
                <input className="form-input" placeholder="John Doe" value={form.name || ''} onChange={e => set('name', e.target.value)} required />
              </div>
              <div className="form-group">
                <label className="form-label">Email</label>
                <input className="form-input" type="email" placeholder="your@email.com" value={form.email || ''} onChange={e => set('email', e.target.value)} required />
              </div>
            </div>
            <div className="form-group">
              <label className="form-label">Password</label>
              <input className="form-input" type="password" placeholder="Min 6 characters" value={form.password || ''} onChange={e => set('password', e.target.value)} required />
            </div>
            <div className="grid-2">
              <div className="form-group">
                <label className="form-label">{role === 'student' ? 'Roll Number' : 'Employee ID'}</label>
                <input className="form-input" placeholder={role === 'student' ? 'e.g. CS2024001' : 'e.g. TCH001'} value={form[role === 'student' ? 'roll_number' : 'employee_id'] || ''} onChange={e => set(role === 'student' ? 'roll_number' : 'employee_id', e.target.value)} required />
              </div>
              <div className="form-group">
                <label className="form-label">Department</label>
                <input className="form-input" placeholder="e.g. Computer Science" value={form.department || ''} onChange={e => set('department', e.target.value)} />
              </div>
            </div>
            {role === 'student' && (
              <div className="form-group">
                <label className="form-label">Current Semester</label>
                <select className="form-select" value={form.current_semester || 1} onChange={e => set('current_semester', parseInt(e.target.value))}>
                  {[1,2,3,4,5,6,7,8].map(s => <option key={s} value={s}>Semester {s}</option>)}
                </select>
              </div>
            )}
            {role === 'teacher' && (
              <div className="form-group">
                <label className="form-label">Subject Specialization</label>
                <input className="form-input" placeholder="e.g. Mathematics, Physics" value={form.subject_specialization || ''} onChange={e => set('subject_specialization', e.target.value)} />
              </div>
            )}
            <div className="form-group">
              <label className="form-label">Phone (Optional)</label>
              <input className="form-input" placeholder="+91 9876543210" value={form.phone || ''} onChange={e => set('phone', e.target.value)} />
            </div>
            <button className="btn btn-primary w-full" type="submit" disabled={loading} style={{ justifyContent: 'center' }}>
              {loading ? <span className="loading-spinner" /> : '📝 Register'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
