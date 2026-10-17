export interface EngineInfo {
  native: boolean;
  cuda: boolean;
  device: string;
}

export interface Strategy {
  key: string;
  label: string;
  dte: number;
  description: string;
  tags: string[];
}

export interface SetupLeg {
  kind: "call" | "put";
  side: "long" | "short";
  strike: number;
  bid: number;
  ask: number;
  mid: number;
  delta: number;
  iv: number;
  open_interest: number | null;
}

export interface Setup {
  strategy: string;
  label: string;
  expiry: string;
  dte: number;
  legs: SetupLeg[];
  legs_desc: string;
  net_premium: number;
  entry_cash: number;
  max_profit: number | null;
  max_loss: number | null;
  pop: number;
  min_oi: number | null;
  spread_pct: number;
  delta_err: number;
  net_delta: number;
  oi_score: number | null;
  spread_score: number;
  delta_score: number;
  score: number;
}

export interface RankResponse {
  ticker: string;
  spot: number;
  expiry: string;
  dte: number;
  source: string;
  setups: Setup[];
}

export interface Trade {
  entry_date: string;
  exit_date: string;
  expiry: string;
  legs: string;
  entry_spot: number;
  exit_spot: number;
  iv: number;
  entry_cash: number;
  exit_cash: number;
  fees: number;
  pnl: number;
  reason: "target" | "stop" | "expiry" | "end";
  days_held: number;
  model_value: number;
  edge: number;
  mc_pop: number;
}

export interface BacktestMetrics {
  n_trades: number;
  total_pnl: number;
  win_rate: number;
  avg_win: number;
  avg_loss: number;
  profit_factor: number | null;
  max_drawdown: number;
  max_drawdown_pct: number;
  cagr: number;
  sharpe: number;
  avg_days_held: number;
}

export interface BacktestResponse {
  metrics: BacktestMetrics;
  engine_backend: string;
  equity: { date: string; equity: number }[];
  trades: Trade[];
}

export interface PriceResponse {
  price: number;
  std_error: number;
  ci_low: number;
  ci_high: number;
  elapsed_ms: number;
  rng_init_ms: number;
  n_paths: number;
  n_steps: number;
  backend: string;
  prob_positive_payoff: number;
  histogram: { lo: number; hi: number; count: number }[];
  payoff_curve: { spot: number; payoff: number }[];
}
