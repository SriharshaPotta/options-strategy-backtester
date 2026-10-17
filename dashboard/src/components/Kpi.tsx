export default function Kpi({ label, value, tone }: { label: string; value: string; tone?: "pos" | "neg" | "" }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className={`kpi-value ${tone ?? ""}`}>{value}</div>
    </div>
  );
}
