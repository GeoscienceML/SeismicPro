"""Generate a test SEG-Y file"""

import pytest

from seismicpro import make_prestack_segy


FILE_NAME = "test_prestack"
N_SAMPLES = 1000
SAMPLE_INTERVAL = 2
DELAY = 0


@pytest.fixture(scope="package", autouse=True)
def segy_path(tmp_path_factory):
    """Create a temporary SEG-Y file with randomly generated traces."""
    path = tmp_path_factory.mktemp("data") / f"{FILE_NAME}.sgy"
    make_prestack_segy(path, survey_size=(300, 300), origin=(0, 0), sources_step=(50, 150), receivers_step=(100, 25),
                       bin_size=(50, 50), activation_dist=(200, 200), n_samples=N_SAMPLES,
                       sample_interval=SAMPLE_INTERVAL * 1000, delay=DELAY, bar=False, trace_gen=None)
    return path
