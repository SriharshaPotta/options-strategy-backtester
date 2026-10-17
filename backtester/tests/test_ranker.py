from datetime import date

import numpy as np
import pandas as pd
import pytest

from osb.chains import Chain, synthetic_chain
from osb.ranker import RankWeights, rank_setups, score

ASOF = date(2024, 1, 2)
EXPIRY = date(2024, 2, 2)


def test_synthetic_chain_shape_and_deltas():
    chain = synthetic_chain("SPY", 470.0, ASOF, EXPIRY, 0.15)
    calls, puts = chain.side("call"), chain.side("put")
    assert not calls.empty and not puts.empty
    assert (calls["delta"].diff().dropna() <= 1e-12).all()  # call delta falls with strike
    assert (chain.frame["ask"] >= chain.frame["bid"]).all()
    assert chain.frame["open_interest"].isna().all()


def test_every_strategy_produces_valid_setups():
    chain = synthetic_chain("SPY", 470.0, ASOF, EXPIRY, 0.15)
    ranked = rank_setups(chain, top_n=None)
    assert set(ranked["strategy"]) >= {"iron_condor", "bull_put_spread", "short_put", "long_straddle"}
    assert ranked["score"].is_monotonic_decreasing
    ic = ranked[ranked["strategy"] == "iron_condor"].iloc[0]
    strikes = [leg["strike"] for leg in ic["legs"]]
    assert strikes == sorted(strikes)
    assert ic["net_premium"] > 0 and ic["max_loss"] < 0 and 0.4 < ic["pop"] < 0.9


def _chain_with(oi: dict[float, float], spread: dict[float, float]) -> Chain:
    strikes = np.array(sorted(oi))
    frame = pd.DataFrame({
        "kind": "put", "strike": strikes, "mid": 2.0,
        "bid": [2.0 - spread[k] for k in strikes], "ask": [2.0 + spread[k] for k in strikes],
        "iv": 0.2, "delta": -0.30, "open_interest": [oi[k] for k in strikes], "volume": 0.0,
    })
    return Chain("XYZ", 100.0, ASOF, EXPIRY, 0.04, frame, source="test")


def test_ranking_prefers_open_interest_and_tight_spreads():
    chain = _chain_with(oi={95.0: 50, 96.0: 5000, 97.0: 5000}, spread={95.0: 0.05, 96.0: 0.05, 97.0: 0.40})
    ranked = rank_setups(chain, ["short_put"], k=3, top_n=None)
    assert list(ranked["legs_desc"]) == ["-96P", "-95P", "-97P"]


def test_missing_open_interest_drops_weight():
    df = pd.DataFrame({"min_oi": [np.nan, np.nan], "spread_pct": [0.05, 0.20], "delta_err": [0.0, 0.0]})
    out = score(df, RankWeights())
    assert out["score"].iloc[0] > out["score"].iloc[1]
    assert out["score"].iloc[0] == pytest.approx(100 * (0.35 * 0.8 + 0.30) / 0.65)
