"""Matplotlib defaults, as in a Jupyter notebook: default fonts, boxed axes, no grid, default colours. Used only for data charts."""
import contextlib, functools
import matplotlib as mpl
import matplotlib.axes, matplotlib.spines
from unittest import mock
NB_BLUE, NB_ORANGE, NB_GREEN, NB_RED, NB_PURPLE, NB_BROWN, NB_GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#7f7f7f"
@contextlib.contextmanager
def nb():
    with mpl.rc_context():
        mpl.rcdefaults()
        with mock.patch.object(mpl.axes.Axes, "grid", lambda self, *a, **k: None), mock.patch.object(mpl.spines.Spine, "set_visible", lambda self, b: None):
            yield
def nbwrap(f):
    @functools.wraps(f)
    def g(*a, **k):
        with nb(): return f(*a, **k)
    return g
