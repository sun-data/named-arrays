import pytest
import matplotlib.pyplot as plt


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """
    Run the tests in ``named_arrays/tests`` before the parametrized families.

    They are few but include the slowest tests in the suite, such as the
    first compilation of the regridding kernels, so scheduling them first
    stops one :mod:`xdist` worker from finishing minutes after the others.
    """
    items.sort(key=lambda item: not item.nodeid.startswith("named_arrays/tests/"))


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
