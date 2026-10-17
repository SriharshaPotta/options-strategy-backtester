const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });
const usd2 = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2 });
const int = new Intl.NumberFormat("en-US");

export const fmtUsd = (v: number | null | undefined) => (v == null ? "—" : usd.format(v));
export const fmtUsd2 = (v: number | null | undefined) => (v == null ? "—" : usd2.format(v));
export const fmtInt = (v: number | null | undefined) => (v == null ? "—" : int.format(Math.round(v)));
export const fmtPct = (v: number | null | undefined, digits = 1) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
export const fmtNum = (v: number | null | undefined, digits = 2) => (v == null ? "—" : v.toFixed(digits));
export const signClass = (v: number | null | undefined) => (v == null || v === 0 ? "" : v > 0 ? "pos" : "neg");
