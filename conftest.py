import pytest
import matplotlib.pyplot as plt


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Start every test with an empty :mod:`matplotlib.pyplot` figure registry.

    Plotting functions fall back to the current axes when none are given,
    so a figure left open by an earlier test would otherwise leak its
    units and artists into later tests, which makes the outcome depend on
    the order in which the tests happen to run.
    """
    plt.close("all")
    yield
    plt.close("all")
