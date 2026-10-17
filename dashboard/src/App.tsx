import { useEffect, useState } from "react";
import { api } from "./api";
import type { EngineInfo, Strategy } from "./types";
import RankedSetups from "./components/RankedSetups";
import Backtest from "./components/Backtest";
import Pricer from "./components/Pricer";

type Tab = "rank" | "backtest" | "pricer";

const TABS: { key: Tab; label: string }[] = [
  { key: "rank", label: "Ranked Setups" },
  { key: "backtest", label: "Backtest" },
  { key: "pricer", label: "MC Pricer" },
];

export default function App() {
  const [tab, setTab] = useState<Tab>("rank");
  const [engine, setEngine] = useState<EngineInfo | null>(null);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [apiError, setApiError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.engine(), api.strategies()])
      .then(([e, s]) => {
        setEngine(e);
        setStrategies(s);
      })
      .catch((err: Error) => setApiError(err.message));
  }, []);

  return (
    <div className="shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">OSB</span>
          <span className="brand-name">Options Strategy Backtester</span>
        </div>
        <nav className="tabs" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.key}
              role="tab"
              aria-selected={tab === t.key}
              className={tab === t.key ? "tab active" : "tab"}
              onClick={() => setTab(t.key)}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <EngineBadge engine={engine} />
      </header>

      {apiError && <div className="banner error">API unreachable: {apiError}. Start it with <code>uvicorn api.main:app</code>.</div>}

      <main>
        {tab === "rank" && <RankedSetups strategies={strategies} />}
        {tab === "backtest" && <Backtest strategies={strategies} />}
        {tab === "pricer" && <Pricer engine={engine} />}
      </main>
    </div>
  );
}

function EngineBadge({ engine }: { engine: EngineInfo | null }) {
  if (!engine) return <span className="badge muted">engine…</span>;
  if (engine.cuda) return <span className="badge ok" title="CUDA Monte Carlo backend">CUDA · {engine.device}</span>;
  if (engine.native) return <span className="badge warn">C++ CPU engine · {engine.cpu_threads} threads</span>;
  return <span className="badge muted">NumPy fallback</span>;
}
