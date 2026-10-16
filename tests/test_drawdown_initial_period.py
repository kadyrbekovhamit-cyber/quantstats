"""Regressions for drawdowns already present in the first observation."""
import pandas as pd
import pytest
from quantstats import reports, stats


def test_unrecovered_initial_drawdown_has_details():
    returns = pd.Series([-0.05, -0.05, 0.02], index=pd.date_range("2024-01-01", periods=3))
    details = stats.drawdown_details(stats.to_drawdown_series(returns))
    assert len(details) == 1
    row = details.iloc[0]
    assert (row["start"], row["valley"], row["end"], row["days"]) == (
        "2024-01-01", "2024-01-02", "2024-01-03", 3)
    # Equity is 0.95, 0.9025, 0.92055 relative to starting equity 1.
    assert row["max drawdown"] == pytest.approx(-9.75)
    assert row["max drawdown"] / 100 == pytest.approx(stats.max_drawdown(returns))


@pytest.mark.parametrize("drawdowns, expected_count", [
    ([-0.1, 0.0, 0.0], 1),
    ([-0.1, 0.0, -0.2, -0.1], 2),
    ([0.0, -0.1, 0.0], 1),
    ([0.0, 0.0, 0.0], 0),
])
def test_initial_period_and_existing_period_controls(drawdowns, expected_count):
    dd = pd.Series(drawdowns, index=pd.date_range("2024-01-01", periods=len(drawdowns)))
    details = stats.drawdown_details(dd)
    assert len(details) == expected_count
    if expected_count and drawdowns[0] < 0:
        assert details.iloc[0]["start"] == "2024-01-01"
        assert details.iloc[0]["end"] == "2024-01-01"


def test_dataframe_keeps_initial_drawdown():
    dd = pd.DataFrame({"initial": [-0.05, -0.0975, -0.07945], "later": [0.0, -0.1, 0.0]},
                      index=pd.date_range("2024-01-01", periods=3))
    details = stats.drawdown_details(dd)
    assert details["initial"].iloc[0]["max drawdown"] == pytest.approx(-9.75)
    assert details["later"].iloc[0]["max drawdown"] == pytest.approx(-10.0)


def test_full_report_keeps_strategy_and_benchmark_drawdowns():
    index = pd.date_range("2024-01-01", periods=6)
    returns = pd.Series([-0.1, 0.05, -0.02, 0.03, -0.01, 0.02], index=index)
    benchmark = pd.Series([0.01, -0.02, 0.03, 0.01, -0.01, 0.02], index=index)
    metrics = reports.metrics(returns, benchmark=benchmark, mode="full", display=False)
    assert float(metrics.loc["Max Drawdown", "Strategy"]) == pytest.approx(-0.1)
    assert str(metrics.loc["Longest DD Days", "Strategy"]) == "6"
    assert str(metrics.loc["Longest DD Days", "Benchmark"]) == "1"
