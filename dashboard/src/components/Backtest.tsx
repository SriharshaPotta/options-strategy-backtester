import { useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import { fmtNum, fmtPct, fmtUsd, signClass } from "../format";
import type { BacktestResponse, Strategy } from "../types";
import Kpi from "./Kpi";

const twoYearsAgo = () => {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 2);
  return d.toISOString().slice(0, 10);
};

export default function Backtest({ strategies }: { strategies: Strategy[] }) {
  const [form, setForm] = useState({
    ticker: "SPY",
    strategy: "iron_condor",
    start: twoYearsAgo(),
    end: "",
    entry_every: 5,
    max_open: 4,
    profit_target: 50,
    stop_loss: 200,
  });
  const [result, setResult] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = <K extends keyof typeof form>(k: K, v: (typeof form)[K]) => setForm((f) => ({ ...f, [k]: v }));

  const run = () => {
    setLoading(true);
    setError(null);
    api
      .backtest({
        ticker: form.ticker,
        strategy: form.strategy,
        start: form.start,
        end: form.end || null,
        entry_every: form.entry_every,
        max_open: form.max_open,
        profit_target: form.profit_target / 100,
        stop_loss: form.stop_loss / 100,
      })
      .then(setResult)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const tradeBars = useMemo(
    () => (result?.trades ?? []).map((t, i) => ({ i: i + 1, pnl: t.pnl, date: t.exit_date, reason: t.reason })),
    [result],
  );
  const byReason = useMemo(() => {
    const m = new Map<string, { n: number; pnl: number }>();
    for (const t of result?.trades ?? []) {
      const cur = m.get(t.reason) ?? { n: 0, pnl: 0 };
      m.set(t.reason, { n: cur.n + 1, pnl: cur.pnl + t.pnl });
    }
    return [...m.entries()];
  }, [result]);

  const m = result?.metrics;

  return (
    <section className="panel">
      <div className="controls">
        <label>
          Ticker
          <input value={form.ticker} onChange={(e) => set("ticker", e.target.value.toUpperCase())} maxLength={8} />
        </label>
        <label>
          Strategy
          <select value={form.strategy} onChange={(e) => set("strategy", e.target.value)}>
            {strategies.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          Start
          <input type="date" value={form.start} onChange={(e) => set("start", e.target.value)} />
        </label>
        <label>
          End
          <input type="date" value={form.end} onChange={(e) => set("end", e.target.value)} />
        </label>
        <label>
          Entry every (days)
          <input type="number" min={1} max={60} value={form.entry_every} onChange={(e) => set("entry_every", +e.target.value)} />
        </label>
        <label>
          Max open
          <input type="number" min={1} max={20} value={form.max_open} onChange={(e) => set("max_open", +e.target.value)} />
        </label>
        <label>
          Take profit %
          <input type="number" min={5} max={500} value={form.profit_target} onChange={(e) => set("profit_target", +e.target.value)} />
        </label>
        <label>
          Stop loss %
          <input type="number" min={10} max={1000} value={form.stop_loss} onChange={(e) => set("stop_loss", +e.target.value)} />
        </label>
        <button className="primary" onClick={run} disabled={loading}>
          {loading ? "Running…" : "Run backtest"}
        </button>
      </div>

      {error && <div className="banner error">{error}</div>}

      {m && result && (
        <>
          <div className="kpis">
            <Kpi label="Total P&L" value={fmtUsd(m.total_pnl)} tone={signClass(m.total_pnl) as "pos" | "neg" | ""} />
            <Kpi label="Win rate" value={fmtPct(m.win_rate)} />
            <Kpi label="Trades" value={String(m.n_trades)} />
            <Kpi label="Profit factor" value={m.profit_factor == null ? "∞" : fmtNum(m.profit_factor)} />
            <Kpi label="Max drawdown" value={fmtUsd(m.max_drawdown)} tone="neg" />
            <Kpi label="Sharpe" value={fmtNum(m.sharpe)} />
            <Kpi label="Avg win / loss" value={`${fmtUsd(m.avg_win)} / ${fmtUsd(m.avg_loss)}`} />
            <Kpi label="Avg days held" value={fmtNum(m.avg_days_held, 1)} />
          </div>

          <div className="charts">
            <figure className="chart">
              <figcaption>Equity (realized + mark-to-market)</figcaption>
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={result.equity} margin={{ top: 8, right: 12, bottom: 0, left: 8 }}>
                  <defs>
                    <linearGradient id="eq" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.35} />
                      <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="date" tick={{ fill: "var(--muted)", fontSize: 11 }} minTickGap={48} />
                  <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} tickFormatter={(v) => fmtUsd(v)} width={72} />
                  <ReferenceLine y={0} stroke="var(--muted)" strokeDasharray="3 3" />
                  <Tooltip contentStyle={tooltipStyle} formatter={(v: number) => [fmtUsd(v), "equity"]} />
                  <Area type="monotone" dataKey="equity" stroke="var(--accent)" strokeWidth={2} fill="url(#eq)" />
                </AreaChart>
              </ResponsiveContainer>
            </figure>

            <figure className="chart">
              <figcaption>P&amp;L per trade</figcaption>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={tradeBars} margin={{ top: 8, right: 12, bottom: 0, left: 8 }}>
                  <CartesianGrid stroke="var(--grid)" vertical={false} />
                  <XAxis dataKey="i" tick={{ fill: "var(--muted)", fontSize: 11 }} minTickGap={16} />
                  <YAxis tick={{ fill: "var(--muted)", fontSize: 11 }} tickFormatter={(v) => fmtUsd(v)} width={64} />
                  <ReferenceLine y={0} stroke="var(--muted)" />
                  <Tooltip
                    contentStyle={tooltipStyle}
                    formatter={(v: number) => [fmtUsd(v), "P&L"]}
                    labelFormatter={(i: number) => `Trade ${i} · ${tradeBars[i - 1]?.date} · ${tradeBars[i - 1]?.reason}`}
                  />
                  <Bar dataKey="pnl">
                    {tradeBars.map((t) => (
                      <Cell key={t.i} fill={t.pnl >= 0 ? "var(--pos)" : "var(--neg)"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </figure>
          </div>

          <div className="reasons">
            {byReason.map(([reason, v]) => (
              <span key={reason} className="reason">
                {reason}: {v.n} trades · <span className={signClass(v.pnl)}>{fmtUsd(v.pnl)}</span>
              </span>
            ))}
            <span className="reason muted">pricing backend: {result.engine_backend}</span>
          </div>

          <div className="table-wrap">
            <table className="grid">
              <thead>
                <tr>
                  <th>Entry</th>
                  <th>Exit</th>
                  <th>Legs</th>
                  <th>Spot in → out</th>
                  <th>IV</th>
                  <th>Premium</th>
                  <th title="Entry premium minus Monte Carlo fair value">MC edge</th>
                  <th>MC POP</th>
                  <th>Exit</th>
                  <th>P&amp;L</th>
                </tr>
              </thead>
              <tbody>
                {[...result.trades].reverse().map((t) => (
                  <tr key={`${t.entry_date}-${t.legs}`}>
                    <td className="mono">{t.entry_date}</td>
                    <td className="mono">{t.exit_date}</td>
                    <td className="mono">{t.legs}</td>
                    <td className="num">
                      {fmtNum(t.entry_spot)} → {fmtNum(t.exit_spot)}
                    </td>
                    <td className="num">{fmtPct(t.iv)}</td>
                    <td className={`num ${signClass(t.entry_cash)}`}>{fmtUsd(t.entry_cash)}</td>
                    <td className={`num ${signClass(t.edge)}`}>{fmtUsd(t.edge)}</td>
                    <td className="num">{fmtPct(t.mc_pop)}</td>
                    <td>
                      <span className={`tag ${t.reason}`}>{t.reason}</span>
                    </td>
                    <td className={`num ${signClass(t.pnl)}`}>{fmtUsd(t.pnl)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {!result && !error && (
        <p className="empty">
          Backtests run on chains reconstructed from price history (realized vol × premium, modelled skew and
          spreads); live open interest is only available for the current chain.
        </p>
      )}
    </section>
  );
}

const tooltipStyle = {
  background: "var(--surface-2)",
  border: "1px solid var(--border)",
  borderRadius: 6,
  color: "var(--text)",
  fontSize: 12,
};
