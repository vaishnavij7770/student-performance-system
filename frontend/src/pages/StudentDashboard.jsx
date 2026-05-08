import React, { useState, useEffect, useRef } from 'react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Legend, Cell
} from 'recharts';
import Sidebar from '../components/Sidebar';
import API from '../utils/api';
import { useAuth } from '../context/AuthContext';

const NAV = [
  { id: 'profile', icon: '👤', label: 'My Profile' },
  { id: 'performance', icon: '📈', label: 'Performance' },
  { id: 'activities', icon: '🏆', label: 'Activities' },
  { id: 'attendance', icon: '📋', label: 'Attendance' },
];

const SEMESTERS = [1, 2, 3, 4, 5, 6, 7, 8];

const GRADE = (pct) => {
  if (pct >= 90) return { grade: 'A+', color: '#22d3a0' };
  if (pct >= 80) return { grade: 'A', color: '#4f8ef7' };
  if (pct >= 70) return { grade: 'B+', color: '#7c5cfc' };
  if (pct >= 60) return { grade: 'B', color: '#f5a623' };
  if (pct >= 50) return { grade: 'C', color: '#f5a623' };
  return { grade: 'F', color: '#f25c5c' };
};

export default function StudentDashboard() {
  const { user, profile, setProfile } = useAuth();
  const [view, setView] = useState('profile');
  const [perfData, setPerfData] = useState(null);
  const [activeSem, setActiveSem] = useState(null);
  const [editMode, setEditMode] = useState(false);
  const [editForm, setEditForm] = useState({});
  const [msg, setMsg] = useState('');
  const [err, setErr] = useState('');
  const [uploading, setUploading] = useState(false);
  const fileRef = useRef();

  const [attendanceData, setAttendanceData] = useState(null);

  useEffect(() => {
    if (view === 'performance' || view === 'activities') fetchPerformance();
    if (view === 'attendance' && profile?.id) fetchAttendance();
  }, [view, profile]);

  const fetchAttendance = () => {
    API.get(`/attendance/student/${profile.id}`).then(r => setAttendanceData(r.data));
  };

  const fetchPerformance = () => {
    API.get('/student/performance').then(r => {
      setPerfData(r.data);
      const sems = Object.keys(r.data.semester_data).map(Number).sort();
      if (sems.length > 0 && !activeSem) setActiveSem(sems[sems.length - 1]);
    });
  };

  const flash = (isErr, msg) => {
    isErr ? setErr(msg) : setMsg(msg);
    setTimeout(() => { setMsg(''); setErr(''); }, 3000);
  };

  const saveProfile = async () => {
    try {
      const r = await API.put('/student/profile', editForm);
      setProfile(r.data.profile);
      setEditMode(false);
      flash(false, 'Profile updated!');
    } catch { flash(true, 'Update failed'); }
  };

  const handlePhotoUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setUploading(true);
    const fd = new FormData();
    fd.append('photo', file);
    try {
      const r = await API.post('/student/photo', fd, { headers: { 'Content-Type': 'multipart/form-data' } });
      setProfile(p => ({ ...p, photo_url: r.data.photo_url }));
      flash(false, 'Photo updated!');
    } catch { flash(true, 'Upload failed'); }
    setUploading(false);
  };

  // Build chart data for active semester
  const semMarks = perfData?.semester_data?.[String(activeSem)]?.marks || [];
  const semActs = perfData?.semester_data?.[String(activeSem)]?.activities || [];

  const barData = semMarks.map(m => ({
    subject: m.subject.length > 10 ? m.subject.slice(0, 10) + '…' : m.subject,
    fullSubject: m.subject,
    percentage: m.percentage,
    obtained: m.marks_obtained,
    total: m.total_marks,
  }));

  const radarData = semMarks.map(m => ({
    subject: m.subject.length > 8 ? m.subject.slice(0, 8) + '…' : m.subject,
    score: m.percentage,
    fullMark: 100,
  }));

  // Overall semester averages for trend chart
  const trendData = perfData ? Object.entries(perfData.semester_data).sort((a, b) => a[0] - b[0]).map(([sem, data]) => {
    const marks = data.marks;
    const avg = marks.length ? marks.reduce((s, m) => s + m.percentage, 0) / marks.length : 0;
    return { semester: `Sem ${sem}`, avg: Math.round(avg * 10) / 10 };
  }) : [];

  const customTooltip = ({ active, payload }) => {
    if (active && payload?.length) {
      const d = payload[0].payload;
      return (
        <div style={{ background: 'var(--card)', border: '1px solid var(--border)', padding: '10px 14px', borderRadius: 9, fontSize: '0.82rem' }}>
          <div style={{ fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>{d.fullSubject || d.subject}</div>
          <div style={{ color: 'var(--text2)' }}>{d.obtained}/{d.total} marks</div>
          <div style={{ color: GRADE(d.percentage).color, fontWeight: 700 }}>{d.percentage}% · {GRADE(d.percentage).grade}</div>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="app-layout">
      <Sidebar items={NAV} active={view} onNav={setView} />
      <main className="main-content">
        {msg && <div className="success-msg mb-4">✅ {msg}</div>}
        {err && <div className="error-msg mb-4">⚠️ {err}</div>}

        {/* PROFILE VIEW */}
        {view === 'profile' && (
          <>
            <div className="page-header">
              <h1>My Profile</h1>
              <p>Manage your personal information and photo</p>
            </div>

            <div className="grid-2" style={{ gap: 20, alignItems: 'start' }}>
              {/* Photo card */}
              <div className="card" style={{ textAlign: 'center', padding: '32px 24px' }}>
                <div style={{ display: 'flex', justifyContent: 'center', marginBottom: 20 }}>
                  <div className="photo-upload-area" onClick={() => fileRef.current?.click()}>
                    <div className="avatar avatar-xl">
                      {profile?.photo_url
                        ? <img src={`http://localhost:5000${profile.photo_url}`} alt="Profile" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        : user?.name?.charAt(0).toUpperCase()
                      }
                    </div>
                    <div className="photo-upload-overlay">
                      {uploading ? <span className="loading-spinner" /> : '📷'}
                    </div>
                  </div>
                  <input ref={fileRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handlePhotoUpload} />
                </div>
                <h2 style={{ fontWeight: 800, fontSize: '1.3rem' }}>{user?.name}</h2>
                <p style={{ color: 'var(--text2)', marginTop: 4 }}>{user?.email}</p>
                <div style={{ marginTop: 12, display: 'flex', justifyContent: 'center', gap: 8, flexWrap: 'wrap' }}>
                  <span className="badge badge-blue">{profile?.department || 'No Dept.'}</span>
                  <span className="badge badge-purple">Sem {profile?.current_semester}</span>
                  <span className="badge badge-green">{profile?.roll_number}</span>
                </div>
                <p style={{ marginTop: 16, fontSize: '0.78rem', color: 'var(--text3)' }}>Click photo to update</p>
              </div>

              {/* Info card */}
              <div className="card">
                <div className="flex-between mb-4">
                  <h3 style={{ fontWeight: 700 }}>Personal Information</h3>
                  {!editMode ? (
                    <button className="btn btn-secondary btn-sm" onClick={() => { setEditMode(true); setEditForm({ name: user?.name, phone: profile?.phone, address: profile?.address, department: profile?.department }); }}>
                      ✏️ Edit
                    </button>
                  ) : (
                    <div className="flex gap-2">
                      <button className="btn btn-success btn-sm" onClick={saveProfile}>💾 Save</button>
                      <button className="btn btn-ghost btn-sm" onClick={() => setEditMode(false)}>Cancel</button>
                    </div>
                  )}
                </div>

                {!editMode ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                    {[
                      ['📛 Full Name', user?.name],
                      ['📧 Email', user?.email],
                      ['🎫 Roll Number', profile?.roll_number],
                      ['🏫 Department', profile?.department || '—'],
                      ['📅 Semester', `Semester ${profile?.current_semester}`],
                      ['📞 Phone', profile?.phone || '—'],
                      ['📍 Address', profile?.address || '—'],
                    ].map(([label, val]) => (
                      <div key={label}>
                        <div style={{ fontSize: '0.72rem', color: 'var(--text3)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 3 }}>{label}</div>
                        <div style={{ fontSize: '0.9rem', color: 'var(--text)' }}>{val}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div>
                    <div className="form-group">
                      <label className="form-label">Full Name</label>
                      <input className="form-input" value={editForm.name || ''} onChange={e => setEditForm(f => ({ ...f, name: e.target.value }))} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Department</label>
                      <input className="form-input" value={editForm.department || ''} onChange={e => setEditForm(f => ({ ...f, department: e.target.value }))} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Phone</label>
                      <input className="form-input" value={editForm.phone || ''} onChange={e => setEditForm(f => ({ ...f, phone: e.target.value }))} />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Address</label>
                      <textarea className="form-textarea" rows={2} value={editForm.address || ''} onChange={e => setEditForm(f => ({ ...f, address: e.target.value }))} />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* PERFORMANCE VIEW */}
        {view === 'performance' && (
          <>
            <div className="page-header">
              <h1>Academic Performance</h1>
              <p>Semester-wise marks and analysis</p>
            </div>

            {trendData.length > 1 && (
              <div className="card mb-4">
                <h3 style={{ fontWeight: 700, marginBottom: 16 }}>📈 Overall Trend</h3>
                <ResponsiveContainer width="100%" height={180}>
                  <BarChart data={trendData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis dataKey="semester" tick={{ fill: 'var(--text2)', fontSize: 12 }} />
                    <YAxis domain={[0, 100]} tick={{ fill: 'var(--text2)', fontSize: 12 }} />
                    <Tooltip contentStyle={{ background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 9, color: 'var(--text)' }} />
                    <Bar dataKey="avg" name="Avg %" fill="var(--accent)" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="semester-pills">
              {SEMESTERS.map(s => {
                const hasData = perfData?.semester_data?.[String(s)]?.marks?.length > 0;
                return (
                  <button key={s} className={`semester-pill ${activeSem === s ? 'active' : ''}`} onClick={() => setActiveSem(s)} style={{ opacity: hasData ? 1 : 0.4 }}>
                    Sem {s} {hasData ? '●' : ''}
                  </button>
                );
              })}
            </div>

            {semMarks.length === 0 ? (
              <div className="card empty-state">
                <div className="icon">📚</div>
                <p>No marks recorded for Semester {activeSem}</p>
              </div>
            ) : (
              <>
                {/* Stats row */}
                <div className="grid-4 mb-4">
                  {[
                    { label: 'Subjects', value: semMarks.length, color: 'var(--accent)' },
                    { label: 'Avg %', value: `${Math.round(semMarks.reduce((s, m) => s + m.percentage, 0) / semMarks.length)}%`, color: 'var(--green)' },
                    { label: 'Highest', value: `${Math.max(...semMarks.map(m => m.percentage))}%`, color: 'var(--accent2)' },
                    { label: 'Lowest', value: `${Math.min(...semMarks.map(m => m.percentage))}%`, color: 'var(--yellow)' },
                  ].map(s => (
                    <div key={s.label} className="stat-card" style={{ textAlign: 'center' }}>
                      <div className="stat-value" style={{ color: s.color, fontSize: '1.6rem' }}>{s.value}</div>
                      <div className="stat-label">{s.label}</div>
                    </div>
                  ))}
                </div>

                <div className="grid-2 mb-4" style={{ gap: 16 }}>
                  {/* Bar chart */}
                  <div className="card">
                    <h3 style={{ fontWeight: 700, marginBottom: 14 }}>Subject-wise Scores</h3>
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart data={barData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                        <XAxis dataKey="subject" tick={{ fill: 'var(--text2)', fontSize: 11 }} />
                        <YAxis domain={[0, 100]} tick={{ fill: 'var(--text2)', fontSize: 11 }} />
                        <Tooltip content={customTooltip} />
                        <Bar dataKey="percentage" name="%" radius={[6, 6, 0, 0]}>
                          {barData.map((entry, i) => (
                            <Cell key={i} fill={GRADE(entry.percentage).color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Radar chart */}
                  {radarData.length >= 3 && (
                    <div className="card">
                      <h3 style={{ fontWeight: 700, marginBottom: 14 }}>Performance Radar</h3>
                      <ResponsiveContainer width="100%" height={220}>
                        <RadarChart data={radarData}>
                          <PolarGrid stroke="var(--border)" />
                          <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--text2)', fontSize: 11 }} />
                          <Radar name="Score" dataKey="score" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.2} />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  )}
                </div>

                {/* Marks table */}
                <div className="card">
                  <h3 style={{ fontWeight: 700, marginBottom: 16 }}>Detailed Results — Semester {activeSem}</h3>
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr><th>Subject</th><th>Marks</th><th>Percentage</th><th>Grade</th><th>Type</th><th>Remarks</th></tr>
                      </thead>
                      <tbody>
                        {semMarks.map(m => {
                          const g = GRADE(m.percentage);
                          return (
                            <tr key={m.id}>
                              <td style={{ color: 'var(--text)', fontWeight: 600 }}>{m.subject}</td>
                              <td>{m.marks_obtained}/{m.total_marks}</td>
                              <td>
                                <div className="perf-bar">
                                  <div className="perf-bar-track" style={{ width: 80 }}>
                                    <div className="perf-bar-fill" style={{ width: `${m.percentage}%`, background: g.color }} />
                                  </div>
                                  <span style={{ color: g.color, fontWeight: 700, fontSize: '0.82rem' }}>{m.percentage}%</span>
                                </div>
                              </td>
                              <td>
                                <span className="badge" style={{ background: `${g.color}22`, color: g.color }}>{g.grade}</span>
                              </td>
                              <td><span className="badge badge-blue">{m.exam_type}</span></td>
                              <td style={{ color: 'var(--text3)', fontSize: '0.8rem' }}>{m.remarks || '—'}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </>
        )}

        {/* ACTIVITIES VIEW */}
        {view === 'activities' && (
          <>
            <div className="page-header">
              <h1>Extracurricular Activities</h1>
              <p>Your achievements beyond academics</p>
            </div>

            <div className="semester-pills">
              <button className={`semester-pill ${!activeSem ? 'active' : ''}`} onClick={() => setActiveSem(null)}>All</button>
              {SEMESTERS.map(s => (
                <button key={s} className={`semester-pill ${activeSem === s ? 'active' : ''}`} onClick={() => setActiveSem(s)}>Sem {s}</button>
              ))}
            </div>

            {(() => {
              const allActs = perfData ? Object.entries(perfData.semester_data).flatMap(([sem, d]) =>
                d.activities.map(a => ({ ...a, sem: parseInt(sem) }))
              ).filter(a => !activeSem || a.sem === activeSem) : [];

              if (allActs.length === 0) return (
                <div className="card empty-state">
                  <div className="icon">🏆</div>
                  <p>No activities recorded yet</p>
                </div>
              );

              const catColors = { sports: 'badge-green', cultural: 'badge-purple', technical: 'badge-blue', social: 'badge-yellow', academic: 'badge-red', other: 'badge-blue' };

              return (
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: 14 }}>
                  {allActs.map(a => (
                    <div key={a.id} className="card" style={{ padding: '18px 20px' }}>
                      <div className="flex-between mb-2">
                        <span className={`badge ${catColors[a.category] || 'badge-blue'}`}>{a.category}</span>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text3)' }}>Sem {a.sem}</span>
                      </div>
                      <h4 style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--text)', marginBottom: 6 }}>{a.title}</h4>
                      {a.achievement && (
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, color: 'var(--green)', fontSize: '0.82rem', fontWeight: 600, marginBottom: 4 }}>
                          🏅 {a.achievement}
                        </div>
                      )}
                      {a.description && <p style={{ fontSize: '0.8rem', color: 'var(--text2)' }}>{a.description}</p>}
                      {a.date && <div style={{ marginTop: 8, fontSize: '0.75rem', color: 'var(--text3)' }}>📅 {new Date(a.date).toLocaleDateString()}</div>}
                      <div style={{ marginTop: 6, fontSize: '0.72rem', color: 'var(--text3)' }}>Added by {a.teacher_name}</div>
                    </div>
                  ))}
                </div>
              );
            })()}
          </>
        )}
        {/* ATTENDANCE VIEW */}
        {view === 'attendance' && (
          <>
            <div className="page-header">
              <h1>My Attendance</h1>
              <p>Subject-wise attendance summary</p>
            </div>
            {!attendanceData ? (
              <div className="page-loader"><div className="loading-spinner" /></div>
            ) : Object.keys(attendanceData.summary).length === 0 ? (
              <div className="card empty-state">
                <div className="icon">📋</div>
                <p>No attendance records yet</p>
              </div>
            ) : (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: 14, marginBottom: 20 }}>
                  {Object.entries(attendanceData.summary).map(([subj, data]) => {
                    const pct = data.percentage;
                    const color = pct >= 75 ? 'var(--green)' : pct >= 50 ? 'var(--yellow)' : 'var(--red)';
                    return (
                      <div key={subj} className="card" style={{ padding: '18px 20px' }}>
                        <div style={{ fontWeight: 700, fontSize: '0.95rem', color: 'var(--text)', marginBottom: 10 }}>{subj}</div>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8, fontSize: '0.8rem' }}>
                          <span style={{ color: 'var(--text2)' }}>{data.present}P · {data.absent}A · {data.late}L</span>
                          <span style={{ color, fontWeight: 800 }}>{pct}%</span>
                        </div>
                        <div style={{ height: 8, background: 'var(--bg3)', borderRadius: 4, overflow: 'hidden' }}>
                          <div style={{ height: '100%', width: `${pct}%`, background: color, borderRadius: 4, transition: 'width 0.8s' }} />
                        </div>
                        {pct < 75 && (
                          <div style={{ marginTop: 8, fontSize: '0.72rem', color: 'var(--red)' }}>
                            ⚠️ Below 75% threshold
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
                <div className="card">
                  <h3 style={{ fontWeight: 700, marginBottom: 14 }}>Recent Records</h3>
                  <div className="table-wrap">
                    <table>
                      <thead><tr><th>Subject</th><th>Date</th><th>Status</th><th>Remarks</th></tr></thead>
                      <tbody>
                        {attendanceData.records.slice(0, 50).map(r => (
                          <tr key={r.id}>
                            <td style={{ color: 'var(--text)', fontWeight: 600 }}>{r.subject}</td>
                            <td>{new Date(r.date).toLocaleDateString()}</td>
                            <td>
                              <span className={`badge ${r.status === 'present' ? 'badge-green' : r.status === 'absent' ? 'badge-red' : 'badge-yellow'}`}>
                                {r.status}
                              </span>
                            </td>
                            <td style={{ color: 'var(--text3)', fontSize: '0.8rem' }}>{r.remarks || '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </>
            )}
          </>
        )}
      </main>
    </div>
  );
}
