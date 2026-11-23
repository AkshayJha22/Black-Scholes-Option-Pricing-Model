"""Black-Scholes pricing, Greeks and implied volatility, including the version 1 bugs."""
import numpy as np
import pytest

from ivsurface.bs import (black_implied_vol, black_price, bs_delta, bs_greeks, bs_price,
                          implied_vol, implied_vol_brent, option_sign, price_bounds)


def test_textbook_reference_value():
    assert bs_price(100, 100, 1.0, 0.05, 0.20, "call") == pytest.approx(10.4506, abs=1e-4)


@pytest.mark.parametrize("K", [60.0, 90.0, 100.0, 125.0, 180.0])
def test_put_call_parity(K):
    S, T, r, q, s = 100.0, 0.75, 0.03, 0.012, 0.27
    c, p = bs_price(S, K, T, r, s, "call", q), bs_price(S, K, T, r, s, "put", q)
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)


def test_bug_a_option_type_is_case_insensitive():
    """Version 1: bs_delta(100,100,1,0.05,0.3,'CALL') returned the put delta, -0.376."""
    assert bs_delta(100, 100, 1, 0.05, 0.3, "CALL") == pytest.approx(0.6242517, abs=1e-6)
    assert bs_delta(100, 100, 1, 0.05, 0.3, "Put") == pytest.approx(0.6242517 - 1, abs=1e-6)
    assert bs_price(100, 100, 1, 0.05, 0.3, "C") == bs_price(100, 100, 1, 0.05, 0.3, "call")


def test_bug_d_invalid_type_raises():
    """Version 1 printed 'Invalid Option Type' and returned None."""
    with pytest.raises(ValueError):
        bs_price(100, 100, 1, 0.05, 0.3, "straddle")
    with pytest.raises(ValueError):
        option_sign("x")


@pytest.mark.parametrize("kind", ["call", "put"])
def test_bug_c_zero_time_and_zero_vol_limits(kind):
    """Version 1 raised ZeroDivisionError; the limits are discounted intrinsic values."""
    w = 1 if kind == "call" else -1
    S, r, q = 100.0, 0.05, 0.02
    for K in (80.0, 100.0, 120.0):
        assert bs_price(S, K, 0.0, r, 0.3, kind, q) == pytest.approx(max(w * (S - K), 0.0))
        T = 0.5
        expected = max(w * (S * np.exp(-q * T) - K * np.exp(-r * T)), 0.0)
        assert bs_price(S, K, T, r, 0.0, kind, q) == pytest.approx(expected)
        assert bs_price(S, K, T, r, 1e-7, kind, q) == pytest.approx(expected, abs=1e-6)
    arr = bs_price(S, np.array([90.0, 110.0]), np.array([0.0, 0.0]), r, 0.3, kind, q)
    assert np.all(np.isfinite(arr))
    g = bs_greeks(S, 90.0, 0.0, r, 0.3, kind, q)
    assert all(np.isfinite(v) for v in g.values())


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("K", [80.0, 100.0, 120.0])
def test_greeks_match_finite_differences(kind, K):
    S, T, r, s, q = 100.0, 0.6, 0.04, 0.25, 0.01
    g = bs_greeks(S, K, T, r, s, kind, q)
    h = 1e-3
    p = lambda **kw: bs_price(kw.get("S", S), K, kw.get("T", T), kw.get("r", r), kw.get("s", s), kind, q)
    assert g["delta"] == pytest.approx((p(S=S + h) - p(S=S - h)) / (2 * h), abs=1e-6)
    assert g["gamma"] == pytest.approx((p(S=S + h) - 2 * p() + p(S=S - h)) / h ** 2, abs=1e-4)
    assert g["vega"] == pytest.approx((p(s=s + 1e-5) - p(s=s - 1e-5)) / 2e-5, rel=1e-5)
    assert g["rho"] == pytest.approx((p(r=r + 1e-6) - p(r=r - 1e-6)) / 2e-6, rel=1e-4)
    assert g["theta"] == pytest.approx(-(p(T=T + 1e-6) - p(T=T - 1e-6)) / 2e-6, rel=1e-4)


def test_implied_vol_round_trip_vectorised():
    K = np.array([50.0, 70.0, 85.0, 100.0, 115.0, 140.0, 200.0])
    for T in (7 / 365, 0.25, 1.0, 2.0):
        for kind in ("call", "put"):
            for s in (0.08, 0.3, 0.9):
                prices = bs_price(100.0, K, T, 0.03, s, kind, 0.01)
                iv = implied_vol(prices, 100.0, K, T, 0.03, kind, 0.01)
                lower, upper = price_bounds(100.0, K, T, 0.03, kind, 0.01)
                # where the time value is too small to resolve, no volatility can be recovered
                has_time_value = (prices - lower > 1e-6) & (upper - prices > 1e-6)
                assert has_time_value.sum() >= 1
                assert np.all(np.isfinite(iv[has_time_value]))
                assert np.allclose(iv[has_time_value], s, atol=1e-6)


def test_implied_vol_agrees_with_brent():
    for K in (80.0, 100.0, 130.0):
        price = float(bs_price(100, K, 0.5, 0.02, 0.33, "put", 0.0))
        assert float(implied_vol(price, 100, K, 0.5, 0.02, "put")) == pytest.approx(
            implied_vol_brent(price, 100, K, 0.5, 0.02, "put"), abs=1e-8)


def test_implied_vol_outside_bounds_is_nan_not_an_error():
    lower, upper = price_bounds(100, 80, 1.0, 0.05, "call")
    assert np.isnan(implied_vol(float(lower) - 0.01, 100, 80, 1.0, 0.05, "call"))
    assert np.isnan(implied_vol(float(upper) + 0.01, 100, 80, 1.0, 0.05, "call"))
    assert np.isnan(implied_vol_brent(0.0, 100, 80, 1.0, 0.05, "call"))


def test_black76_matches_black_scholes_on_the_forward():
    F, T, df = 105.0, 0.5, np.exp(-0.04 * 0.5)
    K = np.array([90.0, 105.0, 120.0])
    c = black_price(F, K, T, 0.22, df, "call")
    expected = df * (F * __import__("scipy.stats", fromlist=["norm"]).norm.cdf(
        (np.log(F / K) + 0.5 * 0.22 ** 2 * T) / (0.22 * np.sqrt(T)))
        - K * __import__("scipy.stats", fromlist=["norm"]).norm.cdf(
        (np.log(F / K) - 0.5 * 0.22 ** 2 * T) / (0.22 * np.sqrt(T))))
    assert np.allclose(c, expected, atol=1e-10)
    assert np.allclose(black_implied_vol(c, F, K, T, df, "call"), 0.22, atol=1e-8)
