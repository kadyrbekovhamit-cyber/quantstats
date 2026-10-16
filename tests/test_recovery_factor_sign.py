"""Recovery factor must retain the sign of excess total returns."""
import numpy as np
import pandas as pd
import pytest
from quantstats import stats


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("rf", [0.0, 0.02])
def test_losing_strategy_has_negative_recovery_factor(as_frame, rf):
    returns = pd.Series([-0.05, -0.05, 0.02], index=pd.date_range("2024-01-01", periods=3))
    values = returns.to_frame("strategy") if as_frame else returns
    actual = stats.recovery_factor(values, rf=rf)
    actual = actual["strategy"] if as_frame else actual
    # Sum of returns is -0.08; the equity trough is 0.95**2 = 0.9025.
    assert actual == pytest.approx((-0.08 - rf) / 0.0975)
    assert actual < 0


def test_mixed_dataframe_preserves_each_column_sign():
    data = pd.DataFrame({"loss": [-0.05, -0.05, 0.02], "gain": [0.1, -0.05, 0.03]},
                        index=pd.date_range("2024-01-01", periods=3))
    actual = stats.recovery_factor(data)
    assert actual["loss"] == pytest.approx(-0.08 / 0.0975)
    assert actual["gain"] == pytest.approx(0.08 / 0.05)


@pytest.mark.parametrize("as_frame", [False, True])
def test_zero_drawdown_still_returns_nan(as_frame):
    values = pd.Series([0.01, 0.02, 0.03], index=pd.date_range("2024-01-01", periods=3))
    values = values.to_frame("strategy") if as_frame else values
    actual = stats.recovery_factor(values)
    actual = actual["strategy"] if as_frame else actual
    assert np.isnan(actual)
