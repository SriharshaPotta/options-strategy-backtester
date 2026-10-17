import { Fragment, useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { fmtInt, fmtNum, fmtPct, fmtUsd, fmtUsd2, signClass } from "../format";
import type { RankResponse, Setup, Strategy } from "../types";

type SortKey = "score" | "net_premium" | "pop" | "min_oi" | "spread_pct" | "delta_err" | "net_delta" | "max_loss";

const COLUMNS: { key: SortKey | null; label: string; title?: string }[] = [
  { key: "score", label: "Score" },
  { key: null, label: "Strategy" },
  { key: null, label: "Legs" },
  { key: "net_premium", label: "Premium", title: "Credit (+) or debit (−) at mid, per contract" },
  { key: "pop", label: "POP", title: "Risk-neutral probability of profit at expiry" },
  { key: "min_oi", label: "Min OI", title: "Smallest open interest across legs" },
  { key: "spread_pct", label: "Spread", title: "Half bid/ask cost as % of net premium" },
  { key: "delta_err", label: "Δ miss", title: "Mean distance from target deltas" },
  { key: "net_delta", label: "Net Δ" },
  { key: "max_loss", label: "Max loss" },
];

export default function RankedSetups({ strategies }: { strategies: Strategy[] }) {
  const [ticker, setTicker] = useState("SPY");
  const [source, setSource] = useState<"live" | "synthetic">("live");
  const [expiries, setExpiries] = useState<string[]>([]);
  const [expiry, setExpiry] = useState<string>("");
  const [selected, setSelected] = useState<string[]>([]);
  const [data, setData] = useState<RankResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: "score", desc: true });
  const [open, setOpen] = useState<number | null>(null);

  useEffect(() => {
    if (source !== "live" || !ticker) return;
    let cancelled = false;
    api
      .expiries(ticker)
      .then((e) => {
        if (cancelled) return;
        setExpiries(e);
        const target = Date.now() + 30 * 86_400_000;
        const nearest = e.reduce(
          (best, x) => (Math.abs(Date.parse(x) - target) < Math.abs(Date.parse(best) - target) ? x : best),
          e[0] ?? "",
        );
        setExpiry(nearest);
      })
      .catch(() => !cancelled && setExpiries([]));
    return () => {
      cancelled = true;
    };
  }, [ticker, source]);

  const run = () => {
    setLoading(true);
    setError(null);
    setOpen(null);
    api
      .rank({ ticker, source, expiry: source === "live" ? expiry || undefined : undefined, strategies: selected })
      .then(setData)
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  };

  const rows = useMemo(() => {
    if (!data) return [];
    const v = (s: Setup) => (s[sort.key] ?? -Infinity) as number;
    return [...data.setups].sort((a, b) => (sort.desc ? v(b) - v(a) : v(a) - v(b)));
  }, [data, sort]);

  const toggle = (key: string) =>
    setSelected((cur) => (cur.includes(key) ? cur.filter((k) => k !== key) : [...cur, key]));

  return (
    <section className="panel">
      <div className="controls">
        <label>
          Ticker
          <input value={ticker} onChange={(e) => setTicker(e.target.value.toUpperCase())} maxLength={8} />
        </label>
        <label>
          Chain
          <select value={source} onChange={(e) => setSource(e.target.value as "live" | "synthetic")}>
            <option value="live">Live (yfinance)</option>
            <option value="synthetic">Synthetic</option>
          </select>
        </label>
        {source === "live" && (
          <label>
            Expiry
            <select value={expiry} onChange={(e) => setExpiry(e.target.value)}>
              {expiries.map((e) => (
                <option key={e} value={e}>
                  {e}
                </option>
              ))}
            </select>
          </label>
        )}
        <button className="primary" onClick={run} disabled={loading}>
          {loading ? "Ranking…" : "Rank setups"}
        </button>
      </div>

      <div className="chips" aria-label="Strategies">
        {strategies.map((s) => (
          <button
            key={s.key}
            className={selected.includes(s.key) ? "chip on" : "chip"}
            onClick={() => toggle(s.key)}
            title={s.description}
          >
            {s.label}
          </button>
        ))}
        <span className="hint">{selected.length ? `${selected.length} selected` : "all strategies"}</span>
      </div>

      {error && <div className="banner error">{error}</div>}

      {data && (
        <>
          <div className="summary-line">
            <strong>{data.ticker}</strong> spot {fmtUsd2(data.spot)} · expiry {data.expiry} ({data.dte} DTE) ·{" "}
            {data.source} chain · {data.setups.length} setups
          </div>
          <div className="table-wrap">
            <table className="grid">
              <thead>
                <tr>
                  {COLUMNS.map((c) => (
                    <th
                      key={c.label}
                      title={c.title}
                      className={c.key ? "sortable" : ""}
                      aria-sort={c.key === sort.key ? (sort.desc ? "descending" : "ascending") : undefined}
                      onClick={() =>
                        c.key && setSort((s) => ({ key: c.key!, desc: s.key === c.key ? !s.desc : true }))
                      }
                    >
                      {c.label}
                      {c.key === sort.key && <span className="arrow">{sort.desc ? "▾" : "▴"}</span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((s, i) => (
                  <Fragment key={`${s.strategy}-${s.legs_desc}`}>
                    <tr className={open === i ? "row open" : "row"} onClick={() => setOpen(open === i ? null : i)}>
                      <td>
                        <div className="score">
                          <span className="score-bar" style={{ width: `${s.score}%` }} />
                          <span className="num">{fmtNum(s.score, 1)}</span>
                        </div>
                      </td>
                      <td>{s.label}</td>
                      <td className="mono">{s.legs_desc}</td>
                      <td className={`num ${signClass(s.net_premium)}`}>{fmtUsd(s.net_premium)}</td>
                      <td className="num">{fmtPct(s.pop)}</td>
                      <td className="num">{fmtInt(s.min_oi)}</td>
                      <td className="num">{fmtPct(s.spread_pct)}</td>
                      <td className="num">{fmtNum(s.delta_err, 3)}</td>
                      <td className={`num ${signClass(s.net_delta)}`}>{fmtNum(s.net_delta, 2)}</td>
                      <td className="num neg">{s.max_loss == null ? "unlimited" : fmtUsd(s.max_loss)}</td>
                    </tr>
                    {open === i && (
                      <tr className="detail">
                        <td colSpan={COLUMNS.length}>
                          <table className="legs">
                            <thead>
                              <tr>
                                <th>Side</th>
                                <th>Type</th>
                                <th>Strike</th>
                                <th>Bid</th>
                                <th>Ask</th>
                                <th>Δ</th>
                                <th>IV</th>
                                <th>OI</th>
                              </tr>
                            </thead>
                            <tbody>
                              {s.legs.map((l) => (
                                <tr key={`${l.kind}${l.strike}${l.side}`}>
                                  <td className={l.side === "long" ? "pos" : "neg"}>{l.side}</td>
                                  <td>{l.kind}</td>
                                  <td className="num">{l.strike}</td>
                                  <td className="num">{fmtNum(l.bid)}</td>
                                  <td className="num">{fmtNum(l.ask)}</td>
                                  <td className="num">{fmtNum(l.delta, 3)}</td>
                                  <td className="num">{fmtPct(l.iv)}</td>
                                  <td className="num">{fmtInt(l.open_interest)}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          <div className="subscores">
                            OI score {fmtNum(s.oi_score)} · spread score {fmtNum(s.spread_score)} · delta score{" "}
                            {fmtNum(s.delta_score)} · max profit {s.max_profit == null ? "unlimited" : fmtUsd(s.max_profit)}
                          </div>
                        </td>
                      </tr>
                    )}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
      {!data && !error && <p className="empty">Pick a ticker and rank the current chain.</p>}
    </section>
  );
}
