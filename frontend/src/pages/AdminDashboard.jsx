import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from '../components/Sidebar';
import API from '../utils/api';

const NAV = [
  { id: 'overview', icon: '📊', label: 'Overview' },
  { id: 'pending', icon: '⏳', label: 'Pending Approvals' },
  { id: 'students', icon: '🎓', label: 'All Students' },
  { id: 'teachers', icon: '👨‍🏫', label: 'All Teachers' },
];

export default function AdminDashboard() {
  const [view, setView] = useState('overview');
  const [stats, setStats] = useState(null);
  const [pending, setPending] = useState([]);
  const [students, setStudents] = useState([]);
  const [teachers, setTeachers] = useState([]);
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState('');

  const fetchStats = useCallback(() => API.get('/admin/stats').then(r => setStats(r.data)), []);
  const fetchPending = useCallback(() => API.get('/admin/pending').then(r => setPending(r.data)), []);
  const fetchStudents = useCallback(() => API.get('/admin/users?role=student').then(r => setStudents(r.data)), []);
  const fetchTeachers = useCallback(() => API.get('/admin/users?role=teacher').then(r => setTeachers(r.data)), []);

  useEffect(() => {
    fetchStats();
    fetchPending();
  }, [fetchStats, fetchPending]);

  useEffect(() => {
    if (view === 'students') fetchStudents();
    if (view === 'teachers') fetchTeachers();
  }, [view, fetchStudents, fetchTeachers]);

  const approve = async (id) => {
    setLoading(true);
    await API.put(`/admin/approve/${id}`);
    setMsg('User approved!');
    fetchPending(); fetchStats();
    setTimeout(() => setMsg(''), 3000);
    setLoading(false);
  };

  const reject = async (id) => {
    if (!window.confirm('Reject and delete this user?')) return;
    setLoading(true);
    await API.delete(`/admin/reject/${id}`);
    setMsg('User rejected.');
    fetchPending(); fetchStats();
    setTimeout(() => setMsg(''), 3000);
    setLoading(false);
  };

  return (
    <div className="app-layout">
      <Sidebar items={NAV} active={view} onNav={setView} />
      <main className="main-content">
        {msg && <div className="success-msg mb-4">✅ {msg}</div>}

        {view === 'overview' && (
          <>
            <div className="page-header">
              <h1>Admin Dashboard</h1>
              <p>Manage students, teachers, and approvals</p>
            </div>
            {stats && (
              <div className="grid-4 mb-6">
                <div className="stat-card blue">
                  <div className="stat-icon">🎓</div>
                  <div className="stat-value" style={{ color: 'var(--accent)' }}>{stats.total_students}</div>
                  <div className="stat-label">Total Students</div>
                </div>
                <div className="stat-card green">
                  <div className="stat-icon">✅</div>
                  <div className="stat-value" style={{ color: 'var(--green)' }}>{stats.approved_students}</div>
                  <div className="stat-label">Approved Students</div>
                </div>
                <div className="stat-card purple">
                  <div className="stat-icon">👨‍🏫</div>
                  <div className="stat-value" style={{ color: 'var(--accent2)' }}>{stats.total_teachers}</div>
                  <div className="stat-label">Total Teachers</div>
                </div>
                <div className="stat-card red">
                  <div className="stat-icon">⏳</div>
                  <div className="stat-value" style={{ color: 'var(--red)' }}>{stats.pending_approvals}</div>
                  <div className="stat-label">Pending Approvals</div>
                </div>
              </div>
            )}
            {pending.length > 0 && (
              <div className="card">
                <div className="flex-between mb-4">
                  <h3 style={{ fontWeight: 700 }}>Recent Pending Approvals</h3>
                  <button className="btn btn-secondary btn-sm" onClick={() => setView('pending')}>View All</button>
                </div>
                <PendingTable users={pending.slice(0, 5)} onApprove={approve} onReject={reject} />
              </div>
            )}
          </>
        )}

        {view === 'pending' && (
          <>
            <div className="page-header">
              <h1>Pending Approvals</h1>
              <p>{pending.length} users awaiting approval</p>
            </div>
            <div className="card">
              {pending.length === 0 ? (
                <div className="empty-state">
                  <div className="icon">🎉</div>
                  <p>No pending approvals!</p>
                </div>
              ) : (
                <PendingTable users={pending} onApprove={approve} onReject={reject} />
              )}
            </div>
          </>
        )}

        {view === 'students' && (
          <>
            <div className="page-header">
              <h1>All Students</h1>
              <p>{students.filter(s => s.is_approved).length} approved · {students.filter(s => !s.is_approved).length} pending</p>
            </div>
            <div className="card">
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Student</th>
                      <th>Roll No.</th>
                      <th>Department</th>
                      <th>Semester</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {students.map(s => (
                      <tr key={s.id}>
                        <td>
                          <div className="flex-center gap-2">
                            <div className="avatar" style={{ width: 32, height: 32, fontSize: '0.75rem' }}>
                              {s.profile?.photo_url ? <img src={`http://localhost:5000${s.profile.photo_url}`} alt="" /> : s.name?.charAt(0)}
                            </div>
                            <div>
                              <div style={{ fontWeight: 600, color: 'var(--text)' }}>{s.name}</div>
                              <div style={{ fontSize: '0.75rem', color: 'var(--text3)' }}>{s.email}</div>
                            </div>
                          </div>
                        </td>
                        <td>{s.profile?.roll_number}</td>
                        <td>{s.profile?.department || '—'}</td>
                        <td>Sem {s.profile?.current_semester}</td>
                        <td>
                          <span className={`badge ${s.is_approved ? 'badge-green' : 'badge-yellow'}`}>
                            {s.is_approved ? 'Approved' : 'Pending'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}

        {view === 'teachers' && (
          <>
            <div className="page-header">
              <h1>All Teachers</h1>
              <p>{teachers.filter(t => t.is_approved).length} approved</p>
            </div>
            <div className="card">
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Teacher</th>
                      <th>Employee ID</th>
                      <th>Department</th>
                      <th>Specialization</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {teachers.map(t => (
                      <tr key={t.id}>
                        <td>
                          <div className="flex-center gap-2">
                            <div className="avatar" style={{ width: 32, height: 32, fontSize: '0.75rem' }}>
                              {t.name?.charAt(0)}
                            </div>
                            <div>
                              <div style={{ fontWeight: 600, color: 'var(--text)' }}>{t.name}</div>
                              <div style={{ fontSize: '0.75rem', color: 'var(--text3)' }}>{t.email}</div>
                            </div>
                          </div>
                        </td>
                        <td>{t.profile?.employee_id}</td>
                        <td>{t.profile?.department || '—'}</td>
                        <td>{t.profile?.subject_specialization || '—'}</td>
                        <td>
                          <span className={`badge ${t.is_approved ? 'badge-green' : 'badge-yellow'}`}>
                            {t.is_approved ? 'Approved' : 'Pending'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

function PendingTable({ users, onApprove, onReject }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Email</th>
            <th>Role</th>
            <th>ID</th>
            <th>Registered</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {users.map(u => (
            <tr key={u.id}>
              <td style={{ color: 'var(--text)', fontWeight: 600 }}>{u.name}</td>
              <td>{u.email}</td>
              <td><span className={`badge ${u.role === 'teacher' ? 'badge-purple' : 'badge-blue'}`}>{u.role}</span></td>
              <td>{u.profile?.roll_number || u.profile?.employee_id || '—'}</td>
              <td>{new Date(u.created_at).toLocaleDateString()}</td>
              <td>
                <div className="flex gap-2">
                  <button className="btn btn-success btn-sm" onClick={() => onApprove(u.id)}>✅ Approve</button>
                  <button className="btn btn-danger btn-sm" onClick={() => onReject(u.id)}>✗ Reject</button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
