"""Survey fixtures with optionally collected stats"""

# pylint: disable=redefined-outer-name
import pytest
import numpy as np

from seismicpro import Survey, make_prestack_segy


@pytest.fixture(params=["TRACE_SEQUENCE_FILE", ("FieldRecord",), ["INLINE_3D", "CROSSLINE_3D"]])
def survey_no_stats(segy_path, request):
    """Return surveys with no stats collected."""
    return Survey(segy_path, header_index=request.param, header_cols="all", n_workers=1, bar=False, validate=False)


@pytest.fixture
def survey_stats(survey_no_stats):
    """Return surveys with collected stats."""
    # copy is needed if both survey_no_stats and survey_stats are accessed by a single test or fixture
    return survey_no_stats.copy().collect_stats()


@pytest.fixture(params=[True, False])
def survey(survey_no_stats, request):
    """Return surveys with and without collected stats."""
    if request.param:
        # copy is needed since otherwise survey_no_stats will be updated inplace and only surveys with collected stats
        # will be returned
        return survey_no_stats.copy().collect_stats()
    return survey_no_stats


def gen_random_traces(n_traces, n_samples):
    """Generate `n_traces` random traces."""
    return np.random.normal(size=(n_traces, n_samples)).astype(np.float32)


def gen_random_traces_some_dead(n_traces, n_samples):
    """Generate `n_traces` random traces with every third of them dead."""
    traces = np.random.uniform(size=(n_traces, n_samples)).astype(np.float32)
    traces[::3] = 0
    return traces


@pytest.fixture(params=[gen_random_traces, gen_random_traces_some_dead], scope="module")
def stat_segy(tmp_path_factory, request):
    """Return a path to a SEG-Y file and its trace data to estimate its statistics."""
    n_traces = 16
    n_samples = 10
    trace_gen = request.param
    trace_data = trace_gen(n_traces, n_samples)

    def gen_trace(TRACE_SEQUENCE_FILE, **kwargs):  # pylint: disable=invalid-name
        """Return a corresponding trace from pregenerated data."""
        _ = kwargs
        return trace_data[TRACE_SEQUENCE_FILE - 1]

    path = tmp_path_factory.mktemp("stat") / "stat.sgy"
    make_prestack_segy(path, survey_size=(4, 4), origin=(0, 0), sources_step=(3, 3), receivers_step=(1, 1),
                       bin_size=(1, 1), activation_dist=(1, 1), n_samples=n_samples, sample_interval=2000, delay=0,
                       bar=False, trace_gen=gen_trace)
    return path, trace_data
