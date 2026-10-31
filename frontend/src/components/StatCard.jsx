export default function StatCard({ label, value, detail, accent = 'blue' }) {
  const accentClass = {
    blue: 'var(--accent-blue)',
    green: 'var(--accent-green)',
    amber: 'var(--accent-amber)',
    red: 'var(--accent-red)',
  }[accent];

  return (
    <div className="stat-card">
      <div className="stat-header">
        <span>{label}</span>
        <span className="dot" style={{ background: accentClass }} />
      </div>
      <div className="stat-value">{value}</div>
      <div className="stat-detail">{detail}</div>
    </div>
  );
}
