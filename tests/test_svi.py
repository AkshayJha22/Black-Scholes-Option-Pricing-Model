"""SVI fitting and static-arbitrage checks against synthetic chains with known answers."""
import numpy as np
import pytest

from ivsurface.chain import build_surface, clean, forward_and_discount
from ivsurface.svi import (SVIParams, butterfly_g, power_law_phi, quoted_butterflies,
                           risk_neutral_density, ssvi_to_raw, svi_total_variance)
from ivsurface.synthetic import synthetic_chain


def match(truth, T):
    return next(p for t, p in truth.items() if abs(t - T) < 1e-9)


def test_ssvi_to_raw_reproduces_ssvi():
    theta, rho = 0.01, -0.6
    phi = power_law_phi(theta)
    k = np.linspace(-0.5, 0.5, 41)
    ssvi = theta / 2 * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho ** 2))
    assert np.allclose(svi_total_variance(k, ssvi_to_raw(theta, rho, phi)), ssvi, atol=1e-14)


def test_density_integrates_to_one():
    p = ssvi_to_raw(0.02, -0.7, power_law_phi(0.02))
    k = np.linspace(-3, 3, 20001)
    assert np.trapezoid(risk_neutral_density(k, 0.5, p), k) == pytest.approx(1.0, abs=2e-3)


@pytest.fixture(scope="module")
def clean_surface():
    chain, truth = synthetic_chain(seed=1, noise_vol=0.0, rel_spread=0.002)
    return build_surface(chain), truth


def test_fit_recovers_generating_parameters_without_noise(clean_surface):
    surface, truth = clean_surface
    assert len(surface.fits) == len(truth)
    for fit in surface.fits.values():
        p = match(truth, fit.T)
        f = fit.params
        assert fit.rmse_vol < 1e-4
        for got, want in ((f.a, p.a), (f.b, p.b), (f.m, p.m), (f.sigma, p.sigma)):
            assert got == pytest.approx(want, rel=0.1, abs=2e-4)
        assert f.rho == pytest.approx(p.rho, abs=0.03)


def test_fit_recovers_the_smile_with_quote_noise():
    chain, truth = synthetic_chain(seed=2, noise_vol=0.002)
    surface = build_surface(chain)
    for fit in surface.fits.values():
        k = np.linspace(*fit.k_range, 41)
        true_vol = np.sqrt(svi_total_variance(k, match(truth, fit.T)) / fit.T)
        assert np.max(np.abs(fit.implied_vol(k) - true_vol)) < 0.004      # 0.4 vol points


def test_forward_and_discount_from_put_call_parity():
    chain, _ = synthetic_chain(seed=1, noise_vol=0.0, rel_spread=0.002, r=0.04, q=0.015)
    c = clean(chain)
    for days, g in c.groupby("days"):
        T = days / 365
        F, df, n = forward_and_discount(g, 100.0)
        assert F == pytest.approx(100 * np.exp((0.04 - 0.015) * T), rel=1e-4)
        assert df == pytest.approx(np.exp(-0.04 * T), abs=1e-4)
        assert n >= 3


def test_clean_surface_has_no_arbitrage(clean_surface):
    surface, _ = clean_surface
    assert surface.summary["butterfly_ok"].all()
    assert all(c["arbitrage_free"] for c in surface.calendar)
    assert surface.summary["quoted_butterfly_arbitrages"].sum() == 0


def test_injected_butterfly_arbitrage_is_flagged():
    chain, truth = synthetic_chain(seed=1, inject="butterfly")
    T30 = sorted(truth)[2]
    assert butterfly_g(np.linspace(-0.3, 0.3, 601), truth[T30]).min() < 0      # the injected smile
    s = build_surface(chain).summary
    row = s[s["days"] == 30].iloc[0]
    assert (not row["butterfly_ok"]) or row["quoted_butterfly_arbitrages"] > 0


def test_injected_calendar_arbitrage_is_flagged():
    surface = build_surface(synthetic_chain(seed=1, inject="calendar")[0])
    bad = [round(c["T2"] * 365) for c in surface.calendar if not c["arbitrage_free"]]
    assert 91 in bad


def test_quoted_butterflies():
    K = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
    C = np.array([11.0, 7.0, 4.0, 2.0, 1.0])                 # convex: no arbitrage
    assert quoted_butterflies(K, C - 0.05, C + 0.05)["executable_arbitrages"] == 0
    C2 = C.copy()
    C2[2] = 6.0                                             # body too rich: buy wings, sell body
    assert quoted_butterflies(K, C2 - 0.05, C2 + 0.05)["executable_arbitrages"] >= 1
