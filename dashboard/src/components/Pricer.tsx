import { useState } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "../api";
import { fmtInt, fmtNum, fmtPct } from "../format";
import type { EngineInfo, PriceResponse } from "../types";
import Kpi from "./Kpi";

interface LegRow {
  kind: "call" | "put" | "stock";
  strike: number;
  qty: number;
}

const PRESETS: Record<string, LegRow[]> = {
  "Long call": [{ kind: "call", strike: 100, qty: 1 }],
  "Iron condor": [
    { kind: "put", strike: 85, qty: 1 },
    { kind: "put", strike: 92, qty: -1 },
    { kind: "call", strike: 108, qty: -1 },
    { kind: "call", strike: 115, qty: 1 },
  ],
  "Bull put spread": [
    { kind: "put", strike: 90, qty: 1 },
    { kind: "put", strike: 96, qty: -1 },
  ],
  Straddle: [
    { kind: "put", strike: 100, qty: 1 },
    { kind: "call", strike: 100, qty: 1 },
  ],
};

export default function Pricer({ engine }: { engine: EngineInfo | null }) {
  const [market, setMarket] = useState({ spot: 100, vol: 0.2, rate: 0.04, maturity: 0.25 });
  const [legs, setLegs] = useState<LegRow[]>(PRESETS["Iron condor"]);
  const [style, setStyle] = useState("european");
  const [barrier, setBarrier] = useState(130);
  const [nPaths, setNPaths] = useState(1 << 20);
  const [nSteps, setNSteps] = useState(252);
  const [backend, setBackend] = useState("auto");
  const [res, setRes] = useState<PriceResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = () => {
    setLoading(true);
    setError(null);
    api
      .price({ ...market, legs, style, barrier, n_paths: nPaths, n_steps: nSteps, backend })
      .then(setRes)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const updateLeg = (i: number, patch: Partial<LegRow>) =>
    setLegs((ls) => ls.map((l, j) => (j === i ? { ...l, ...patch } : l)));

  const chartData =
    res?.histogram.map((b, i) => ({
      spot: (b.lo + b.hi) / 2,
      count: b.count,
      payoff: (res.payoff_curve[i].payoff + res.payoff_curve[i + 1].payoff) / 2,
    })) ?? [];

  return (
    <section className="panel">
      <div className="controls">
        {(["spot", "vol", "rate", "maturity"] as const).map((k) => (
          <label key={k}>
            {k === "maturity" ? "T (years)" : k}
            <input
              type="number"
              step={k === "spot" ? 1 : 0.01}
              value={market[k]}
              onChange={(e) => setMarket((m) => ({ ...m, [k]: +e.target.value }))}
            />
          </label>
        ))}
        <label>
          Style
          <select value={style} onChange={(e) => setStyle(e.target.value)}>
            <option value="european">European</option>
            <option value="asian">Asian (arith.)</option>
            <option value="up_and_out">Up-and-out</option>
            <option value="down_and_out">Down-and-out</option>
          </select>
        </label>
        {style.endsWith("_out") && (
          <label>
            Barrier
            <input type="number" value={barrier} onChange={(e) => setBarrier(+e.target.value)} />
          </label>
        )}
        <label>
          Paths
          <select value={nPaths} onChange={(e) => setNPaths(+e.target.value)}>
            {[1 << 16, 1 << 18, 1 << 20, 1 << 21, 1 << 22].map((n) => (
              <option key={n} value={n}>
                {fmtInt(n)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Steps
          <input type="number" min={1} max={2000} value={nSteps} onChange={(e) => setNSteps(+e.target.value)} />
        </label>
        <label>
          Backend
          <select value={backend} onChange={(e) => setBackend(e.target.value)}>
            <option value="auto">auto</option>
            <option value="cuda" disabled={!engine?.cuda}>
              cuda
            </option>
            <option value="cpu_mt">cpu ({engine?.cpu_threads ?? "all"} threads)</option>
            <option value="cpu">cpu (1 thread)</option>
          </select>
        </label>
        <button className="primary" onClick={run} disabled={loading || legs.length === 0}>
          {loading ? "Simulating…" : "Price"}
        </button>
      </div>

      <div className="legs-editor">
        <div className="chips">
          {Object.keys(PRESETS).map((p) => (
            <button key={p} className="chip" onClick={() => setLegs(PRESETS[p])}>
              {p}
            </button>
          ))}
          <button
            className="chip"
            disabled={legs.length >= 8}
            onClick={() => setLegs((ls) => [...ls, { kind: "call", strike: market.spot, qty: 1 }])}
          >
            + leg
          </button>
        </div>
        {legs.map((l, i) => (
          <div className="leg-row" key={i}>
            <select value={l.kind} onChange={(e) => updateLeg(i, { kind: e.target.value as LegRow["kind"] })}>
              <option value="call">call</option>
              <option value="put">put</option>
              <option value="stock">stock</option>
            </select>
            <input type="number" value={l.strike} disabled={l.kind === "stock"} onChange={(e) => updateLeg(i, { strike: +e.target.value })} />
            <input type="number" step={1} value={l.qty} onChange={(e) => updateLeg(i, { qty: +e.target.value })} />
            <button className="icon" aria-label="Remove leg" onClick={() => setLegs((ls) => ls.filter((_, j) => j !== i))}>
              ×
            </button>
          </div>
        ))}
      </div>

      {error && <div className="banner error">{error}</div>}

      {res && (
        <>
          <div className="kpis">
            <Kpi label="MC price" value={fmtNum(res.price, 4)} />
            <Kpi label="95% CI" value={`${fmtNum(res.ci_low, 3)} – ${fmtNum(res.ci_high, 3)}`} />
            <Kpi label="Std error" value={fmtNum(res.std_error, 5)} />
            <Kpi label="P(payoff > 0)" value={fmtPct(res.prob_positive_payoff)} />
            <Kpi label="Paths × steps" value={`${fmtInt(res.n_paths)} × ${res.n_steps}`} />
            <Kpi label={`Sim time (${res.backend}${res.threads > 1 ? ` × ${res.threads}` : ""})`} value={`${fmtNum(res.elapsed_ms, 1)} ms`} />
            {res.backend === "cuda" && <Kpi label="RNG init" value={`${fmtNum(res.rng_init_ms, 2)} ms`} />}
          </div>
          <figure className="chart">
            <figcaption>Terminal price distribution and expiry payoff</figcaption>
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart data={chartData} margin={{ top: 8, right: 12, bottom: 0, left: 8 }}>
                <CartesianGrid stroke="var(--grid)" vertical={false} />
                <XAxis dataKey="spot" tick={{ fill: "var(--muted)", fontSize: 11 }} tickFormatter={(v) => fmtNum(v, 0)} />
                <YAxis yAxisId="c" tick={{ fill: "var(--muted)", fontSize: 11 }} tickFormatter={(v) => fmtInt(v)} width={64} />
                <YAxis yAxisId="p" orientation="right" tick={{ fill: "var(--muted)", fontSize: 11 }} width={48} />
                <ReferenceLine yAxisId="p" y={0} stroke="var(--muted)" strokeDasharray="3 3" />
                <Tooltip
                  contentStyle={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text)", fontSize: 12 }}
                  labelFormatter={(v: number) => `S(T) ≈ ${fmtNum(v)}`}
                />
                <Bar yAxisId="c" dataKey="count" fill="var(--accent-soft)" name="paths" />
                <Line yAxisId="p" dataKey="payoff" stroke="var(--accent)" dot={false} strokeWidth={2} name="payoff" />
              </ComposedChart>
            </ResponsiveContainer>
          </figure>
        </>
      )}
    </section>
  );
}
