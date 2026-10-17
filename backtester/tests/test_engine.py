import numpy as np
import pytest

from osb import engine
from osb.greeks import bs_price


def test_european_call_matches_black_scholes():
    r = engine.price(100, 0.05, 0.2, 1.0, [("call", 100, 1)], n_paths=200_000, n_steps=8, seed=3, backend="cpu")
    ref = bs_price(100, 100, 1.0, 0.05, 0.2, kind="call")
    assert abs(r["price"] - ref) < 3 * r["std_error"]
    assert r["backend"] in {"cpu", "numpy"}


def test_terminal_output_mean():
    r = engine.price(100, 0.05, 0.2, 1.0, [("put", 90, 1)], n_paths=100_000, n_steps=1, backend="cpu",
                     return_terminal=True)
    assert r["terminal"].shape == (100_000,)
    assert float(np.mean(r["terminal"])) == pytest.approx(100 * np.exp(0.05), rel=0.01)


def test_payoff_at_expiry_iron_condor():
    legs = [("put", 90, 1), ("put", 95, -1), ("call", 105, -1), ("call", 110, 1)]
    s = np.array([80, 93, 100, 107, 120])
    np.testing.assert_allclose(engine.payoff_at_expiry(legs, s), [-5, -2, 0, -2, -5])


@pytest.mark.skipif(not engine.native_available(), reason="native module not built")
def test_native_path_dependent_styles():
    vanilla = bs_price(100, 100, 1.0, 0.05, 0.2, kind="call")
    asian = engine.price(100, 0.05, 0.2, 1.0, [("call", 100, 1)], style="asian", n_paths=100_000, n_steps=64,
                         backend="cpu")
    ko = engine.price(100, 0.05, 0.2, 1.0, [("call", 100, 1)], style="up_and_out", barrier=130,
                      n_paths=100_000, n_steps=64, backend="cpu")
    assert 0 < asian["price"] < vanilla
    assert 0 < ko["price"] < vanilla


@pytest.mark.skipif(not engine.has_cuda(), reason="no CUDA device")
def test_cuda_agrees_with_cpu():
    legs = [("put", 90, 1), ("put", 95, -1), ("call", 105, -1), ("call", 110, 1)]
    g = engine.price(100, 0.04, 0.2, 0.25, legs, n_paths=1 << 20, n_steps=32, backend="cuda")
    c = engine.price(100, 0.04, 0.2, 0.25, legs, n_paths=1 << 19, n_steps=32, backend="cpu")
    assert abs(g["price"] - c["price"]) < 3 * np.hypot(g["std_error"], c["std_error"])


@pytest.mark.skipif(not engine.native_available(), reason="native module not built")
def test_cpu_mt_matches_single_thread_statistically():
    legs = [("put", 95, -1), ("put", 90, 1)]
    one = engine.price(100, 0.04, 0.2, 0.5, legs, n_paths=200_000, n_steps=16, backend="cpu")
    mt = engine.price(100, 0.04, 0.2, 0.5, legs, n_paths=200_000, n_steps=16, backend="cpu_mt", threads=4)
    assert mt["backend"] == "cpu_mt" and mt["threads"] == 4
    assert abs(one["price"] - mt["price"]) < 3 * np.hypot(one["std_error"], mt["std_error"])
