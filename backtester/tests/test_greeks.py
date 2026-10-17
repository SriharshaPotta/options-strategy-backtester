import numpy as np
import pytest

from osb.greeks import bs_greeks, bs_price, strike_for_delta


def test_known_prices():
    assert bs_price(100, 100, 1.0, 0.05, 0.2, kind="call") == pytest.approx(10.4506, abs=1e-4)
    assert bs_price(100, 100, 1.0, 0.05, 0.2, kind="put") == pytest.approx(5.5735, abs=1e-4)


def test_put_call_parity_vectorised():
    k = np.linspace(70, 130, 13)
    c = bs_price(100, k, 0.5, 0.03, 0.25, 0.01, "call")
    p = bs_price(100, k, 0.5, 0.03, 0.25, 0.01, "put")
    np.testing.assert_allclose(c - p, 100 * np.exp(-0.01 * 0.5) - k * np.exp(-0.03 * 0.5), atol=1e-10)


def test_greeks_signs_and_values():
    g = bs_greeks(100, 100, 1.0, 0.05, 0.2, kind="call")
    assert g["delta"] == pytest.approx(0.6368, abs=1e-4)
    assert g["gamma"] > 0 and g["vega"] > 0 and g["theta"] < 0
    gp = bs_greeks(100, 100, 1.0, 0.05, 0.2, kind="put")
    assert gp["delta"] == pytest.approx(0.6368 - 1.0, abs=1e-4)


def test_expiry_returns_intrinsic():
    assert bs_price(105, 100, 0.0, 0.05, 0.2, kind="call") == pytest.approx(5.0)
    assert bs_price(105, 100, 0.0, 0.05, 0.2, kind="put") == pytest.approx(0.0)


@pytest.mark.parametrize("kind,target", [("call", 0.30), ("put", 0.16), ("call", 0.50), ("put", 0.45)])
def test_strike_for_delta_roundtrip(kind, target):
    k = strike_for_delta(400, 30 / 365, 0.04, 0.18, target, kind)
    delta = bs_greeks(400, k, 30 / 365, 0.04, 0.18, kind=kind)["delta"]
    assert abs(delta) == pytest.approx(target, abs=1e-6)
