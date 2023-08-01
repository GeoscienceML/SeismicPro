"""Test Gather loading methods"""

# pylint: disable=redefined-outer-name
import pytest
import numpy as np

from seismicpro import Survey, make_prestack_segy


@pytest.fixture(scope="module", params=[  # Data type of trace amplitudes and maximum absolute value
    [1, 0.1],
    [1, 1],
    [1, 100],
    [1, 100000000],
    [5, 0.1],
    [5, 1],
    [5, 100],
    [5, 100000000],
])
def load_segy(tmp_path_factory, request):
    """Return a path to a SEG-Y file and its trace data to test data loading."""
    n_traces = 16
    n_samples = 20
    segy_fmt, max_amplitude = request.param
    trace_data = np.random.uniform(-max_amplitude, max_amplitude, size=(n_traces, n_samples)).astype(np.float32)

    def gen_trace(TRACE_SEQUENCE_FILE, **kwargs):  # pylint: disable=invalid-name
        """Return a corresponding trace from pregenerated data."""
        _ = kwargs
        return trace_data[TRACE_SEQUENCE_FILE - 1]

    path = tmp_path_factory.mktemp("load") / "load.sgy"
    make_prestack_segy(path, fmt=segy_fmt, survey_size=(4, 4), origin=(0, 0), sources_step=(3, 3),
                       receivers_step=(1, 1), bin_size=(1, 1), activation_dist=(1, 1), n_samples=n_samples,
                       sample_interval=2000, delay=0, bar=False, trace_gen=gen_trace)
    return path, trace_data


TRACES_POS = [  # Ordinal number of traces to load
    [0],  # Single trace
    [1, 5, 3, 2],  # Multiple traces
    [6, 6, 10, 10, 10],  # Duplicated traces
    np.arange(16),  # All traces
]


@pytest.mark.parametrize("limits", [slice(None), slice(2, 15, 5), slice(10), slice(-5, None)])
class TestLoadTraces:
    """Test trace loading methods."""

    @pytest.mark.parametrize("engine", ["segyio", "memmap"])
    @pytest.mark.parametrize("traces_pos", TRACES_POS)
    def test_load_traces(self, load_segy, limits, engine, traces_pos):
        """Compare loaded traces with the actual ones."""
        path, trace_data = load_segy
        survey = Survey(path, header_index="TRACE_SEQUENCE_FILE", engine=engine, bar=False)

        # load_limits take priority over init_limits
        trace_data = trace_data[traces_pos, limits]
        loaded_data = survey.loader.load_traces(traces_pos, limits=limits)
        assert np.allclose(loaded_data, trace_data)

    @pytest.mark.parametrize("traces_pos", [[7], [11, 13]])
    def test_load_traces_after_mmap_reconstruction(self, load_segy, limits, traces_pos):
        """Compare loaded traces with the actual ones after the memory map is reconstructed."""
        path, trace_data = load_segy
        survey = Survey(path, header_index="TRACE_SEQUENCE_FILE", bar=False)

        # The number of traces will change after filter. Memory map is reconstructed after copy and must remember
        # original data shape.
        survey = survey.filter(lambda tsf: tsf % 2 == 1, "TRACE_SEQUENCE_FILE", inplace=True).copy()

        # load_limits take priority over init_limits
        trace_data = trace_data[traces_pos, limits]
        loaded_data = survey.loader.load_traces(traces_pos, limits=limits)
        assert np.allclose(loaded_data, trace_data)


@pytest.mark.parametrize("init_limits", [slice(None), slice(2, 15, 5), slice(10)])
@pytest.mark.parametrize("load_limits", [None, slice(None, None, 2), slice(-5, None)])
class TestLoadGather:
    """Test `Gather` loading methods."""

    @pytest.mark.parametrize("traces_pos", TRACES_POS)
    def test_load_gather(self, load_segy, init_limits, load_limits, traces_pos):
        """Test gather loading by its headers."""
        path, trace_data = load_segy
        survey = Survey(path, header_index="FieldRecord", limits=init_limits, bar=False)

        gather_headers = survey.headers.iloc[traces_pos]
        gather = survey.load_gather(gather_headers, limits=load_limits)

        # load_limits take priority over init_limits
        limits = init_limits if load_limits is None else load_limits
        gather_data = trace_data[traces_pos, limits]

        assert gather.headers.equals(gather_headers)
        assert np.allclose(gather.data, gather_data)
        assert np.allclose(gather.samples, survey.file_samples[limits])

    def test_get_gather(self, load_segy, init_limits, load_limits):
        """Test gather loading by its index."""
        path, trace_data = load_segy
        survey = Survey(path, header_index="FieldRecord", limits=init_limits, bar=False)
        index = np.random.choice(survey.indices)
        gather = survey.get_gather(index, limits=load_limits)

        assert gather.index == index

        gather_headers = survey.get_headers_by_indices([index])
        assert gather.headers.equals(gather_headers)

        # load_limits take priority over init_limits
        limits = init_limits if load_limits is None else load_limits
        traces_pos = gather["TRACE_SEQUENCE_FILE"] - 1
        gather_data = trace_data[traces_pos, limits]

        assert np.allclose(gather.data, gather_data)
        assert np.allclose(gather.samples, survey.file_samples[limits])

    def test_sample_gather(self, load_segy, init_limits, load_limits):
        """Test gather sampling."""
        path, trace_data = load_segy
        survey = Survey(path, header_index="FieldRecord", limits=init_limits, bar=False)
        gather = survey.sample_gather(limits=load_limits)

        assert gather.index is not None  # Unique index

        gather_headers = survey.get_headers_by_indices([gather.index])
        assert gather.headers.equals(gather_headers)

        # load_limits take priority over init_limits
        limits = init_limits if load_limits is None else load_limits
        traces_pos = gather["TRACE_SEQUENCE_FILE"] - 1
        gather_data = trace_data[traces_pos, limits]

        assert np.allclose(gather.data, gather_data)
        assert np.allclose(gather.samples, survey.file_samples[limits])
