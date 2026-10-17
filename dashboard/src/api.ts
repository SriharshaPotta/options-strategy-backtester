import type { BacktestResponse, EngineInfo, PriceResponse, RankResponse, Strategy } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  engine: () => request<EngineInfo>("/api/engine"),
  strategies: () => request<Strategy[]>("/api/strategies"),
  expiries: (ticker: string) => request<string[]>(`/api/expiries?ticker=${encodeURIComponent(ticker)}`),
  rank: (params: { ticker: string; expiry?: string; strategies: string[]; source: "live" | "synthetic" }) => {
    const q = new URLSearchParams({ ticker: params.ticker, source: params.source, top_n: "50" });
    if (params.expiry) q.set("expiry", params.expiry);
    params.strategies.forEach((s) => q.append("strategy", s));
    return request<RankResponse>(`/api/rank?${q}`);
  },
  backtest: (body: Record<string, unknown>) =>
    request<BacktestResponse>("/api/backtest", { method: "POST", body: JSON.stringify(body) }),
  price: (body: Record<string, unknown>) =>
    request<PriceResponse>("/api/price", { method: "POST", body: JSON.stringify(body) }),
};
