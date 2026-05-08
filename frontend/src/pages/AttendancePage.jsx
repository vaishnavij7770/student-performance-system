import React, { useState, useEffect } from 'react';
import API from '../utils/api';

const SEMESTERS = [1, 2, 3, 4, 5, 6, 7, 8];

export default function AttendancePage() {
  const [students, setStudents] = useState([]);
  const [subject, setSubject] = useState('');
  const [semester, setSemester] = useState(1);
  const [attDate, setAttDate] = useState(new Date().toISOString().split('T')[0]);
  const [records, setRecords] = useState({});   // { student_id: 'present'|'absent'|'late' }
  const [remarks, setRemarks] = useState({});
  const [msg, setMsg] = useState('');
  const [err, setErr] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [report, setReport] = useState([]);
  const [tab, setTab] = useState('mark');

  useEffect(() => {
    API.get(`/teacher/students?semester=${semester}`)
      .then(r => {
        setStudents(r.data);
        const init = {};
        r.data.forEach(s => { init[s.id] = 'present'; });
        setRecords(init);
      });
  }, [semester]);

  const flash = (isErr, m) => {
    isErr ? setErr(m) : setMsg(m);
    setTimeout(() => { setMsg(''); setErr(''); }, 3000);
  };

  const markAll = (status) => {
    const updated = {};
    students.forEach(s => { updated[s.id] = status; });
    setRecords(updated);
  };

  const submitAttendance = async () => {
    if (!subject.trim()) return flash(true, 'Please enter a subject');
    setSubmitting(true);
    try {
      const payload = {
        subject,
        semester,
        date: attDate,
        records: students.map(s => ({
          student_id: s.id,
          status: records[s.id] || 'present',
          remarks: remarks[s.id] || null,
        }))
      };
      await API.post('/attendance/bulk', payload);
      flash(false, `Attendance saved for ${students.length} students!`);
    } catch (e) {
      flash(true, e.response?.data?.error || 'Failed to save');
    }
    setSubmitting(false);
  };

  const fetchReport = async () => {
    try {
      const r = await API.get(`/attendance/report?semester=${semester}`);
      setReport(r.data);
    } catch { flash(true, 'Failed to load report'); }
  };

  useEffect(() => {
    if (tab === 'report') fetchReport();
  }, [tab, semester]);

  const statusColor = { present: 'var(--green)', absent: 'var(--red)', late: 'var(--yellow)' };
  const statusBadge = { present: 'badge-green', absent: 'badge-red', late: 'badge-yellow' };

  // Summary from report
  const reportByDate = report.reduce((acc, r) => {
    const key = `${r.date} — ${r.subject}`;
    if (!acc[key]) acc[key] = { date: r.date, subject: r.subject, present: 0, absent: 0, late: 0 };
    acc[key][r.status]++;
    return acc;
  }, {});

  return (
    <div>
      <div className="page-header">
        <h1>Attendance</h1>
        <p>Mark and track student attendance</p>
      </div>

      <div className="tabs">
        <button className={`tab-btn ${tab === 'mark' ? 'active' : ''}`} onClick={() => setTab('mark')}>✏️ Mark Attendance</button>
        <button className={`tab-btn ${tab === 'report' ? 'active' : ''}`} onClick={() => setTab('report')}>📊 Attendance Report</button>
      </div>

      {msg && <div className="success-msg mb-4">✅ {msg}</div>}
      {err && <div className="error-msg mb-4">⚠️ {err}</div>}

      {tab === 'mark' && (
        <>
          {/* Controls */}
          <div className="card mb-4">
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16 }}>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Subject</label>
                <input className="form-input" placeholder="e.g. Mathematics" value={subject} onChange={e => setSubject(e.target.value)} />
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Semester</label>
                <select className="form-select" value={semester} onChange={e => setSemester(parseInt(e.target.value))}>
                  {SEMESTERS.map(s => <option key={s} value={s}>Semester {s}</option>)}
                </select>
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Date</label>
                <input className="form-input" type="date" value={attDate} onChange={e => setAttDate(e.target.value)} />
              </div>
            </div>

            <div className="flex gap-2 mt-3">
              <button className="btn btn-success btn-sm" onClick={() => markAll('present')}>✅ All Present</button>
              <button className="btn btn-danger btn-sm" onClick={() => markAll('absent')}>❌ All Absent</button>
              <button className="btn btn-secondary btn-sm" onClick={() => markAll('late')}>⏰ All Late</button>
            </div>
          </div>

          {students.length === 0 ? (
            <div className="card empty-state">
              <div className="icon">👥</div>
              <p>No students in Semester {semester}</p>
            </div>
          ) : (
            <div className="card">
              <div className="flex-between mb-4">
                <h3 style={{ fontWeight: 700 }}>
                  {students.length} Students — Semester {semester}
                </h3>
                <div className="flex gap-2">
                  <span className="badge badge-green">{Object.values(records).filter(v => v === 'present').length} Present</span>
                  <span className="badge badge-red">{Object.values(records).filter(v => v === 'absent').length} Absent</span>
                  <span className="badge badge-yellow">{Object.values(records).filter(v => v === 'late').length} Late</span>
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                {students.map((s, i) => (
                  <div key={s.id} style={{
                    display: 'flex', alignItems: 'center', gap: 14,
                    padding: '12px 14px', borderRadius: 10,
                    background: 'var(--bg3)', border: `1px solid ${statusColor[records[s.id]] || 'var(--border)'}22`,
                    transition: 'border-color 0.2s'
                  }}>
                    <span style={{ width: 28, color: 'var(--text3)', fontSize: '0.8rem', fontWeight: 700 }}>
                      {String(i + 1).padStart(2, '0')}
                    </span>
                    <div className="avatar" style={{ width: 34, height: 34, fontSize: '0.8rem', flexShrink: 0 }}>
                      {s.photo_url
                        ? <img src={`http://localhost:5000${s.photo_url}`} alt="" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                        : s.name?.charAt(0)
                      }
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 600, color: 'var(--text)', fontSize: '0.875rem' }}>{s.name}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text3)' }}>{s.roll_number}</div>
                    </div>

                    {/* Status buttons */}
                    <div className="flex gap-2">
                      {['present', 'absent', 'late'].map(status => (
                        <button key={status} onClick={() => setRecords(r => ({ ...r, [s.id]: status }))}
                          style={{
                            padding: '5px 12px', borderRadius: 7, border: 'none', cursor: 'pointer',
                            fontSize: '0.78rem', fontWeight: 700, textTransform: 'capitalize',
                            background: records[s.id] === status
                              ? status === 'present' ? 'var(--green)' : status === 'absent' ? 'var(--red)' : 'var(--yellow)'
                              : 'var(--bg)',
                            color: records[s.id] === status
                              ? status === 'late' ? '#000' : '#fff'
                              : 'var(--text3)',
                            transition: 'all 0.15s',
                          }}>
                          {status === 'present' ? '✓' : status === 'absent' ? '✗' : '⏰'} {status}
                        </button>
                      ))}
                    </div>

                    <input
                      className="form-input"
                      placeholder="Remark..."
                      style={{ width: 140, padding: '6px 10px', fontSize: '0.8rem' }}
                      value={remarks[s.id] || ''}
                      onChange={e => setRemarks(r => ({ ...r, [s.id]: e.target.value }))}
                    />
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 20, display: 'flex', justifyContent: 'flex-end' }}>
                <button className="btn btn-primary" onClick={submitAttendance} disabled={submitting}>
                  {submitting ? <span className="loading-spinner" /> : '💾 Save Attendance'}
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {tab === 'report' && (
        <>
          <div className="flex gap-3 mb-4" style={{ alignItems: 'center' }}>
            <select className="form-select" style={{ width: 180 }} value={semester} onChange={e => setSemester(parseInt(e.target.value))}>
              {SEMESTERS.map(s => <option key={s} value={s}>Semester {s}</option>)}
            </select>
            <button className="btn btn-secondary btn-sm" onClick={fetchReport}>🔄 Refresh</button>
          </div>

          {Object.keys(reportByDate).length === 0 ? (
            <div className="card empty-state">
              <div className="icon">📋</div>
              <p>No attendance records for Semester {semester}</p>
            </div>
          ) : (
            <>
              {/* Summary cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: 12, marginBottom: 20 }}>
                {Object.entries(reportByDate).sort((a, b) => b[0].localeCompare(a[0])).map(([key, data]) => {
                  const total = data.present + data.absent + data.late;
                  const pct = total ? Math.round(((data.present + data.late) / total) * 100) : 0;
                  return (
                    <div key={key} className="card card-sm">
                      <div className="flex-between mb-2">
                        <span style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text)' }}>{data.subject}</span>
                        <span style={{ fontSize: '0.72rem', color: 'var(--text3)' }}>{new Date(data.date).toLocaleDateString()}</span>
                      </div>
                      <div className="flex gap-2" style={{ marginBottom: 8 }}>
                        <span className="badge badge-green">{data.present} Present</span>
                        <span className="badge badge-red">{data.absent} Absent</span>
                        {data.late > 0 && <span className="badge badge-yellow">{data.late} Late</span>}
                      </div>
                      <div style={{ height: 6, background: 'var(--bg)', borderRadius: 3, overflow: 'hidden' }}>
                        <div style={{ height: '100%', width: `${pct}%`, background: pct >= 75 ? 'var(--green)' : 'var(--red)', borderRadius: 3, transition: 'width 0.6s' }} />
                      </div>
                      <div style={{ marginTop: 4, fontSize: '0.72rem', color: 'var(--text3)' }}>{pct}% attendance rate</div>
                    </div>
                  );
                })}
              </div>

              {/* Detailed table */}
              <div className="card">
                <h3 style={{ fontWeight: 700, marginBottom: 16 }}>Detailed Records</h3>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr><th>Student</th><th>Subject</th><th>Date</th><th>Status</th><th>Remarks</th></tr>
                    </thead>
                    <tbody>
                      {report.slice(0, 100).map(r => (
                        <tr key={r.id}>
                          <td style={{ color: 'var(--text)', fontWeight: 600 }}>{r.student_name}</td>
                          <td>{r.subject}</td>
                          <td>{new Date(r.date).toLocaleDateString()}</td>
                          <td><span className={`badge ${statusBadge[r.status]}`}>{r.status}</span></td>
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
    </div>
  );
}
