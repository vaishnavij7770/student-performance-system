import React from 'react';
import { useAuth } from '../context/AuthContext';

export default function Sidebar({ items, active, onNav }) {
  const { user, logout } = useAuth();

  return (
    <div className="sidebar">
      <div className="sidebar-logo">
        <h2>🎓 StudPer</h2>
        <p>{user?.role?.charAt(0).toUpperCase() + user?.role?.slice(1)} Portal</p>
      </div>

      <nav className="sidebar-nav">
        {items.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${active === item.id ? 'active' : ''}`}
            onClick={() => onNav(item.id)}
          >
            <span className="icon">{item.icon}</span>
            {item.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className="flex-center gap-3 mb-4" style={{ padding: '8px 4px' }}>
          <div className="avatar">
            {user?.name?.charAt(0).toUpperCase()}
          </div>
          <div>
            <div style={{ fontSize: '0.85rem', fontWeight: 600 }}>{user?.name}</div>
            <div style={{ fontSize: '0.72rem', color: 'var(--text3)' }}>{user?.email}</div>
          </div>
        </div>
        <button className="btn btn-ghost w-full btn-sm" onClick={logout}>
          🚪 Sign Out
        </button>
      </div>
    </div>
  );
}
