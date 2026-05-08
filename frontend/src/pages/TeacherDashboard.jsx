import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from '../components/Sidebar';
import API from '../utils/api';
import AttendancePage from './AttendancePage';

const NAV = [
  { id: 'students', icon: '👥', label: 'Students' },
  { id: 'marks', icon: '📝', label: 'Add Marks' },
  { id: 'activities', icon: '🏆', label: 'Add Activity' },
  { id: 'attendance', icon: '📋', label: 'Attendance' },
];

const SEMESTERS = [1, 2, 3, 4, 5, 6, 7, 8];

export default function TeacherDashboard() {
  const [view, setView] = useState('students');
  const [students, setStudents] = useState([]);
  const [filterSem, setFilterSem] = useState(null);
  const [selectedStudent, setSelectedStudent] = useState(null);
  const [markForm, setMarkForm] = useState({});
  const [actForm, setActForm] = useState({});
  const [studentMarks, setStudentMarks] = useState([]);
  const [studentActs, setStudentActs] = useState([]);
  const [msg, setMsg] = useState('');
  const [err, setErr] = useState('');

  const setM = (k, v) => setMarkForm(f => ({ ...f, [k]: v }));
  const setA = (k, v) => setActForm(f => ({ ...f, [k]: v }));

  const fetchStudents = useCallback(() => {
    const params = filterSem ? `?semester=${filterSem}` : '';
    API.get(`/teacher/students${params}`).then(r => setStudents(r.data));
  }, [filterSem]);

  useEffect(() => { fetchStudents(); }, [fetchStudents]);

  const selectStudent = async (student) => {
    setSelectedStudent(student);
    setMarkForm({ student_id: student.id, semester: student.current_semester });
    setActForm({ student_id: student.id, semester: student.current_semester });
    const [mRes, aRes] = await Promise.all([
      API.get(`/teacher/marks/${student.id}`),
      API.get(`/teacher/activities/${student.id}`)
    ]);
    setStudentMarks(mRes.data);
    setStudentActs(aRes.data);
  };

  const flash = (isErr, message) => {
    isErr ? setErr(message) : setMsg(message);
    setTimeout(() => { setMsg(''); setErr(''); }, 3000);
  };

  const submitMark = async (e) => {
    e.preventDefault();
    try {
      await API.post('/teacher/marks', { ...markForm, marks_obtained: parseFloat(markForm.marks_obtained), total_marks: parseFloat(markForm.total_marks || 100), semester: parseInt(markForm.semester) });
      flash(false, 'Mark added successfully!');
      if (selectedStudent) {
        const r = await API.get(`/teacher/marks/${selectedStudent.id}`);
        setStudentMarks(r.data);
      }
      setMarkForm(f => ({ student_id: f.student_id, semester: f.semester }));
    } catch (e) { flash(true, e.response?.data?.error || 'Failed to add mark'); }
  };

  const submitActivity = async (e) => {
    e.preventDefault();
    try {
      await API.post('/teacher/activities', { ...actForm, semester: parseInt(actForm.semester) });
      flash(false, 'Activity added!');
      if (selectedStudent) {
        const r = await API.get(`/teacher/activities/${selectedStudent.id}`);
        setStudentActs(r.data);
      }
      setActForm(f => ({ student_id: f.student_id, semester: f.semester }));
    } catch (e) { flash(true, e.response?.data?.error || 'Failed to add activity'); }
  };

  const deleteMark = async (id) => {
    await API.delete(`/teacher/marks/${id}`);
    setStudentMarks(prev => prev.filter(m => m.id !== id));
    flash(false, 'Mark deleted');
  };

  // Group students by semester
  const grouped = students.reduce((acc, s) => {
    const key = `Semester ${s.current_semester}`;
    if (!acc[key]) acc[key] = [];
    acc[key].push(s);
    return acc;
  }, {});

  return (
    <div className="app-layout">
      <Sidebar items={NAV} active={view} onNav={v => { setView(v); setSelectedStudent(null); }} />
      <main className="main-content">
        {msg && <div className="success-msg mb-4">✅ {msg}</div>}
        {err && <div className="error-msg mb-4">⚠️ {err}</div>}

        {view === 'students' && (
          <>
            <div className="page-header">
              <h1>Students</h1>
              <p>All approved students — sorted by semester</p>
            </div>
            <div className="semester-pills mb-4">
              <button className={`semester-pill ${!filterSem ? 'active' : ''}`} onClick={() => setFilterSem(null)}>All Semesters</button>
              {SEMESTERS.map(s => (
                <button key={s} className={`semester-pill ${filterSem === s ? 'active' : ''}`} onClick={() => setFilterSem(s)}>Sem {s}</button>
              ))}
            </div>

            {Object.keys(grouped).sort().map(semKey => (
              <div key={semKey} className="card mb-4">
                <div className="flex-between mb-4">
                  <h3 style={{ fontWeight: 700 }}>{semKey}</h3>
                  <span className="badge badge-blue">{grouped[semKey].length} students</span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Student</th>
                        <th>Roll No.</th>
                        <th>Department</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {grouped[semKey].map(s => (
                        <tr key={s.id}>
                          <td>
                            <div className="flex-center gap-2">
                              <div className="avatar" style={{ width: 34, height: 34, fontSize: '0.8rem' }}>
                                {s.photo_url ? <img src={`http://localhost:5000${s.photo_url}`} alt="" /> : s.name?.charAt(0)}
                              </div>
                              <span style={{ color: 'var(--text)', fontWeight: 600 }}>{s.name}</span>
                            </div>
                          </td>
                          <td>{s.roll_number}</td>
                          <td>{s.department || '—'}</td>
                          <td>
                            <button className="btn btn-secondary btn-sm" onClick={() => { setSelectedStudent(s); selectStudent(s); setView('marks'); }}>
                              📝 Add Marks
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ))}
            {students.length === 0 && (
              <div className="card empty-state">
                <div className="icon">👥</div>
                <p>No approved students yet</p>
              </div>
            )}
          </>
        )}

        {view === 'marks' && (
          <>
            <div className="page-header">
              <h1>Add Marks</h1>
              <p>Select a student and enter their marks</p>
            </div>
            <div className="grid-2" style={{ gap: 20 }}>
              <div>
                {/* Student selector */}
                <div className="card mb-4">
                  <h3 style={{ fontWeight: 700, marginBottom: 14 }}>Select Student</h3>
                  <div style={{ maxHeight: 280, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {students.map(s => (
                      <button key={s.id} onClick={() => selectStudent(s)} style={{
                        display: 'flex', alignItems: 'center', gap: 10, padding: '10px 12px',
                        borderRadius: 9, border: `1px solid ${selectedStudent?.id === s.id ? 'var(--accent)' : 'var(--border)'}`,
                        background: selectedStudent?.id === s.id ? 'rgba(79,142,247,0.1)' : 'var(--bg3)',
                        color: 'var(--text)', cursor: 'pointer', textAlign: 'left', width: '100%',
                      }}>
                        <div className="avatar" style={{ width: 32, height: 32, fontSize: '0.75rem' }}>
                          {s.name?.charAt(0)}
                        </div>
                        <div>
                          <div style={{ fontSize: '0.875rem', fontWeight: 600 }}>{s.name}</div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text3)' }}>{s.roll_number} · Sem {s.current_semester}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Mark form */}
                {selectedStudent && (
                  <div className="card">
                    <h3 style={{ fontWeight: 700, marginBottom: 16 }}>Add Mark for {selectedStudent.name}</h3>
                    <form onSubmit={submitMark}>
                      <div className="grid-2">
                        <div className="form-group">
                          <label className="form-label">Subject</label>
                          <input className="form-input" placeholder="Mathematics" value={markForm.subject || ''} onChange={e => setM('subject', e.target.value)} required />
                        </div>
                        <div className="form-group">
                          <label className="form-label">Semester</label>
                          <select className="form-select" value={markForm.semester || 1} onChange={e => setM('semester', e.target.value)}>
                            {SEMESTERS.map(s => <option key={s} value={s}>Semester {s}</option>)}
                          </select>
                        </div>
                        <div className="form-group">
                          <label className="form-label">Marks Obtained</label>
                          <input className="form-input" type="number" min="0" step="0.5" placeholder="75" value={markForm.marks_obtained || ''} onChange={e => setM('marks_obtained', e.target.value)} required />
                        </div>
                        <div className="form-group">
                          <label className="form-label">Total Marks</label>
                          <input className="form-input" type="number" min="1" placeholder="100" value={markForm.total_marks || ''} onChange={e => setM('total_marks', e.target.value)} required />
                        </div>
                      </div>
                      <div className="form-group">
                        <label className="form-label">Exam Type</label>
                        <select className="form-select" value={markForm.exam_type || 'final'} onChange={e => setM('exam_type', e.target.value)}>
                          <option value="final">Final</option>
                          <option value="midterm">Midterm</option>
                          <option value="assignment">Assignment</option>
                          <option value="quiz">Quiz</option>
                        </select>
                      </div>
                      <div className="form-group">
                        <label className="form-label">Remarks (Optional)</label>
                        <textarea className="form-textarea" rows={2} placeholder="Good performance..." value={markForm.remarks || ''} onChange={e => setM('remarks', e.target.value)} />
                      </div>
                      <button className="btn btn-primary w-full" type="submit" style={{ justifyContent: 'center' }}>
                        ✅ Save Mark
                      </button>
                    </form>
                  </div>
                )}
              </div>

              <div>
                {selectedStudent && (
                  <div className="card">
                    <h3 style={{ fontWeight: 700, marginBottom: 16 }}>
                      📋 Marks History — {selectedStudent.name}
                    </h3>
                    {studentMarks.length === 0 ? (
                      <div className="empty-state" style={{ padding: 24 }}>
                        <p>No marks added yet</p>
                      </div>
                    ) : (
                      <div className="table-wrap">
                        <table>
                          <thead>
                            <tr>
                              <th>Subject</th>
                              <th>Sem</th>
                              <th>Score</th>
                              <th>%</th>
                              <th>Type</th>
                              <th></th>
                            </tr>
                          </thead>
                          <tbody>
                            {studentMarks.map(m => (
                              <tr key={m.id}>
                                <td style={{ color: 'var(--text)', fontWeight: 600 }}>{m.subject}</td>
                                <td>{m.semester}</td>
                                <td>{m.marks_obtained}/{m.total_marks}</td>
                                <td>
                                  <span style={{ color: m.percentage >= 75 ? 'var(--green)' : m.percentage >= 50 ? 'var(--yellow)' : 'var(--red)', fontWeight: 700 }}>
                                    {m.percentage}%
                                  </span>
                                </td>
                                <td><span className="badge badge-blue">{m.exam_type}</span></td>
                                <td>
                                  <button className="btn btn-danger btn-sm" onClick={() => deleteMark(m.id)}>🗑</button>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {view === 'activities' && (
          <>
            <div className="page-header">
              <h1>Add Extracurricular Activity</h1>
              <p>Record students' achievements and activities</p>
            </div>
            <div className="grid-2" style={{ gap: 20 }}>
              <div>
                <div className="card mb-4">
                  <h3 style={{ fontWeight: 700, marginBottom: 14 }}>Select Student</h3>
                  <div style={{ maxHeight: 280, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {students.map(s => (
                      <button key={s.id} onClick={() => selectStudent(s)} style={{
                        display: 'flex', alignItems: 'center', gap: 10, padding: '10px 12px',
                        borderRadius: 9, border: `1px solid ${selectedStudent?.id === s.id ? 'var(--accent)' : 'var(--border)'}`,
                        background: selectedStudent?.id === s.id ? 'rgba(79,142,247,0.1)' : 'var(--bg3)',
                        color: 'var(--text)', cursor: 'pointer', textAlign: 'left', width: '100%',
                      }}>
                        <div className="avatar" style={{ width: 32, height: 32, fontSize: '0.75rem' }}>{s.name?.charAt(0)}</div>
                        <div>
                          <div style={{ fontSize: '0.875rem', fontWeight: 600 }}>{s.name}</div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text3)' }}>{s.roll_number}</div>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>

                {selectedStudent && (
                  <div className="card">
                    <h3 style={{ fontWeight: 700, marginBottom: 16 }}>Activity for {selectedStudent.name}</h3>
                    <form onSubmit={submitActivity}>
                      <div className="form-group">
                        <label className="form-label">Activity Title</label>
                        <input className="form-input" placeholder="District Level Chess Championship" value={actForm.title || ''} onChange={e => setA('title', e.target.value)} required />
                      </div>
                      <div className="grid-2">
                        <div className="form-group">
                          <label className="form-label">Category</label>
                          <select className="form-select" value={actForm.category || 'other'} onChange={e => setA('category', e.target.value)}>
                            {['sports', 'cultural', 'technical', 'social', 'academic', 'other'].map(c => <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>)}
                          </select>
                        </div>
                        <div className="form-group">
                          <label className="form-label">Semester</label>
                          <select className="form-select" value={actForm.semester || 1} onChange={e => setA('semester', e.target.value)}>
                            {SEMESTERS.map(s => <option key={s} value={s}>Semester {s}</option>)}
                          </select>
                        </div>
                      </div>
                      <div className="form-group">
                        <label className="form-label">Achievement</label>
                        <input className="form-input" placeholder="1st Place / Participated / Gold Medal" value={actForm.achievement || ''} onChange={e => setA('achievement', e.target.value)} />
                      </div>
                      <div className="form-group">
                        <label className="form-label">Date</label>
                        <input className="form-input" type="date" value={actForm.date || ''} onChange={e => setA('date', e.target.value)} />
                      </div>
                      <div className="form-group">
                        <label className="form-label">Description</label>
                        <textarea className="form-textarea" rows={2} placeholder="Brief description..." value={actForm.description || ''} onChange={e => setA('description', e.target.value)} />
                      </div>
                      <button className="btn btn-primary w-full" type="submit" style={{ justifyContent: 'center' }}>
                        🏆 Save Activity
                      </button>
                    </form>
                  </div>
                )}
              </div>

              <div>
                {selectedStudent && (
                  <div className="card">
                    <h3 style={{ fontWeight: 700, marginBottom: 16 }}>Activity History — {selectedStudent.name}</h3>
                    {studentActs.length === 0 ? (
                      <div className="empty-state" style={{ padding: 24 }}><p>No activities yet</p></div>
                    ) : (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                        {studentActs.map(a => (
                          <div key={a.id} style={{ padding: '12px 14px', background: 'var(--bg3)', borderRadius: 10, border: '1px solid var(--border)' }}>
                            <div className="flex-between">
                              <span style={{ fontWeight: 700, fontSize: '0.875rem', color: 'var(--text)' }}>{a.title}</span>
                              <span className="badge badge-purple">{a.category}</span>
                            </div>
                            {a.achievement && <div style={{ marginTop: 4, fontSize: '0.8rem', color: 'var(--green)' }}>🏅 {a.achievement}</div>}
                            <div style={{ marginTop: 4, fontSize: '0.75rem', color: 'var(--text3)' }}>Sem {a.semester} · {a.date || 'No date'}</div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
        {view === 'attendance' && <AttendancePage />}
      </main>
    </div>
  );
}
