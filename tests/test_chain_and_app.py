"""The bundled SPY snapshot, the calculator path and the Streamlit app."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from ivsurface.calculator import option_through_time, synthetic_path
from ivsurface.chain import DATA_DIR, build_surface, clean, implied_vols, load_snapshot
from tests.conftest import ROOT

HAS_SNAPSHOT = (DATA_DIR / "SPY_chain_snapshot.csv").exists()


@pytest.mark.skipif(not HAS_SNAPSHOT, reason="no snapshot")
def test_real_spy_surface_fits():
    chain, meta = load_snapshot("SPY")
    s = build_surface(chain).summary
    assert len(s) >= 20
    assert s["rmse_vol_pts"].median() < 0.6
    atm30 = float(np.interp(30, s["days"], s["atm_iv"]))
    assert 0.05 < atm30 < 0.8
    assert (s["skew_25d_put_minus_call"].dropna() > 0).mean() > 0.9     # equity put skew
    assert s["butterfly_ok"].mean() > 0.8


@pytest.mark.skipif(not HAS_SNAPSHOT, reason="no snapshot")
def test_real_spy_discounting_uses_the_tbill():
    """SPY options are American: discount at the T-bill, never at a factor above 1."""
    chain, meta = load_snapshot("SPY")
    ivs = implied_vols(clean(chain))
    per = ivs.groupby("expiry").agg(T=("T", "first"), df=("df", "first"), F=("forward", "first"))
    assert (per["df"] <= 1).all()
    assert np.allclose(per["df"], np.exp(-meta["tbill_13w"] * per["T"]))
    assert ((per["F"] / meta["spot"] - 1).abs() < 0.05).all()
    assert per["F"].is_monotonic_increasing or (per["F"].diff().dropna() > -1.0).all()


def test_bug_b_option_ages_towards_expiry():
    """Version 1's chart had time to expiry RISING along the chart."""
    path = option_through_time(synthetic_path(), None, 0.5, 0.04, 0.25, "call")
    assert np.all(np.diff(path["T"].values) <= 0)
    assert path["T"].iloc[0] == pytest.approx(0.5, abs=3 / 365)
    assert path.attrs["strike"] == pytest.approx(path["spot"].iloc[0])     # bought at the money
    assert np.all(path["option"] >= path["intrinsic"] - 1e-9)
    if path["T"].iloc[-1] == 0:
        assert path["option"].iloc[-1] == pytest.approx(path["intrinsic"].iloc[-1])


@pytest.mark.parametrize("source", ["snapshot", "synthetic"])
def test_app_renders(source):
    if source == "snapshot" and not HAS_SNAPSHOT:
        pytest.skip("no snapshot")
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    if source != "snapshot":
        at.radio[0].set_value(source).run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [m.label for m in at.metric]
    assert "Expiries fitted" in labels and "Call" in labels
