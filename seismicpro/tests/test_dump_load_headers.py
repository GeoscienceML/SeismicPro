"""Test methods for dump and load headers"""
# pylint: disable=too-many-arguments, redefined-outer-name

import pytest
import numpy as np
import pandas as pd

from seismicpro import Survey
from seismicpro.utils import dump_dataframe, load_dataframe

from .survey.asserters import assert_surveys_equal
from .test_gather import assert_gathers_equal


def add_columns(container, size):
    """Add new columns to provided container."""
    def _gen_col(vmin, vmax, dtype, size):
        int_part = np.random.randint(vmin, vmax, size=size)
        if dtype == "float":
            return int_part * np.random.random(size=size)
        return int_part

    ranges = [[0, 1], [-1, 1], [0, 1000], [-1000, 1000], [-1000000, 1000000]]
    for vmin, vmax in ranges:
        for dtype in ["int", "float"]:
            container[f"{dtype}_{np.abs(vmin)}_{vmax}"] = _gen_col(vmin, vmax, dtype, size)
    return container


@pytest.fixture(params=["survey", "gather"])
def containers(segy_path, request):
    """Return two containers: container unchanged and container with additional int and float columns."""
    survey = Survey(segy_path, header_index="FieldRecord", header_cols="all", n_workers=1, bar=False, validate=False)
    container = survey.sample_gather() if request.param == "gather" else survey
    container_with_cols = add_columns(container.copy(), container.n_traces)
    return container, container_with_cols


@pytest.fixture(scope="module")
def dataframe():
    """Return `pd.DataFrame` with 10 numerical columns."""
    df_dict = {}
    return pd.DataFrame(add_columns(df_dict, 100))


@pytest.mark.parametrize("headers_to_dump,headers_to_load,usecols,has_header,float_precision,decimal,sep", [
    [["int_0_1000"], ["int_0_1000"], None, True, 2, ".", ","],
    [["float_1_1"], ["float_1_1"], None, False, 3, ".", ";"],
    [["int_0_1000", "float_0_1000", "int_1000_1000", "float_1000_1000"],
     ["int_0_1000", "float_0_1000", "int_1000_1000", "float_1000_1000"], None, True, 3, ",", ","],
    [["int_0_1000", "float_0_1000", "int_1000_1000", "float_1000_1000"], None, [1, 2], True, 5, ".", ","],
    [["int_0_1000", "float_0_1000", "int_1000_1000", "float_1000_1000"], ["float_0_1000", "int_1000_1000"], [1, 2],
     False, 5, ",", ";"],
    [["float_0_1", "float_1_1", "int_1000000_1000000", "float_1000000_1000000"],
     ["float_1_1", "float_1000000_1000000"], None, True, 3, ".", ";"],
])
@pytest.mark.parametrize("format", ["fwf", "csv"])
def test_dump_load_dataframe(tmp_path, dataframe, headers_to_dump, headers_to_load, usecols, has_header,
                             float_precision, decimal, sep, format):
    """Check that dump_dataframe and load_dataframe works with different ranges of int or float numbers."""
    file_path = tmp_path / "tmp"

    kwargs = {"decimal": decimal} if format == "fwf" else {"sep": sep}
    dump_dataframe(df=dataframe[headers_to_dump], path=file_path, has_header=has_header, format=format,
                   float_precision=float_precision, **kwargs)

    loaded_df = load_dataframe(path=file_path, has_header=has_header, columns=headers_to_load, usecols=usecols,
                               format=format, **kwargs)
    assert_headers = headers_to_load
    if assert_headers is None:
        assert_headers = headers_to_dump if usecols is None else np.array(headers_to_dump)[usecols]
    assert ((dataframe[assert_headers] - loaded_df).max() <= 10**(-float_precision)).all()


@pytest.mark.parametrize("headers_to_dump,headers_to_load,usecols,has_header,float_precision,decimal,sep", [
    [["TRACE_SEQUENCE_FILE", "int_0_1000"], ["TRACE_SEQUENCE_FILE", "int_0_1000"], None, True, 2, ".", ","],
    [["TRACE_SEQUENCE_FILE", "float_1000_1000"], ["TRACE_SEQUENCE_FILE", "float_1000_1000"], None, False, 2, ",", ","],
    [["FieldRecord", "TraceNumber", "int_1000_1000", "float_1_1", "float_0_1000"],
     ["FieldRecord", "TraceNumber", "int_1000_1000", "float_1_1", "float_0_1000"], None, True, 4, ".", ";"],
    [["FieldRecord", "TraceNumber", "int_0_1000", "float_0_1", "float_1000_1000"],
     ["FieldRecord", "TraceNumber", "int_0_1000", "float_0_1", "float_1000_1000"], None, False, 4, ".", ";"],
    [["FieldRecord", "TraceNumber", "SourceX", "SourceY", "int_0_1", "float_0_1", "float_1000_1000"],
     ["FieldRecord", "TraceNumber", "int_0_1", "float_0_1", "float_1000_1000"], None, True, 2, ",", ","],
    [["FieldRecord", "TraceNumber", "SourceX", "SourceY", "float_1000_1000"],
     ["FieldRecord", "TraceNumber", "float_1000_1000"], [0, 1, -1], False, 2, ",", ";"],
    [["FieldRecord", "TraceNumber", "SourceX", "SourceY", "float_1000_1000"], None, [0, 1, -1], True, 2, ".", ","],
    [["FieldRecord", "TraceNumber", "SourceX", "SourceY", "int_1000000_1000000", "float_1000000_1000000"],
     None, None, True, 2, ".", ","],
])
@pytest.mark.parametrize("format", ["fwf", "csv"])
class TestContainers:
    """Test dump and load containers"""

    def test_dump_container(self, tmp_path, containers, headers_to_dump, headers_to_load, usecols, has_header,
                            float_precision, decimal, sep, format):
        """Dump containers headers via `dump_headers` and check that loaded headers via `load_dataframe` has not
        changed."""
        file_path = tmp_path / "tmp"
        _, dump_container = containers

        kwargs = {"decimal": decimal} if format == "fwf" else {"sep": sep}
        dump_container.dump_headers(path=file_path, headers_names=headers_to_dump, dump_headers_names=has_header,
                                    format=format, float_precision=float_precision, **kwargs)

        loaded_df = load_dataframe(path=file_path, has_header=has_header, columns=headers_to_load, usecols=usecols,
                                   format=format, **kwargs)

        assert_headers = headers_to_load
        if assert_headers is None:
            assert_headers = headers_to_dump if usecols is None else np.array(headers_to_dump)[usecols]
        headers = dump_container.get_headers(assert_headers)
        assert ((headers - loaded_df).max() <= 10**(-float_precision)).all()

    def test_dump_load_container(self, tmp_path, containers, headers_to_dump, headers_to_load, usecols, has_header,
                                 float_precision, decimal, sep, format):
        """Dump containers headers via `dump_headers` and check that loaded container via `load_headers` has not
        changed."""
        file_path = tmp_path / "tmp"
        load_container, dump_container = containers

        kwargs = {"decimal": decimal} if format == "fwf" else {"sep": sep}
        dump_container.dump_headers(path=file_path, headers_names=headers_to_dump, dump_headers_names=has_header,
                                    format=format, float_precision=float_precision, **kwargs)

        loaded_container = load_container.load_headers(path=file_path, has_header=has_header,
                                                       headers_names=headers_to_load, usecols=usecols, format=format,
                                                       **kwargs)

        correct_headers = dump_container.headers[loaded_container.headers.columns]
        assert ((correct_headers - loaded_container.headers).max() <= 10**(-float_precision)).all()

        # Since loaded columns haven't changed it is not needed to check them during containers check.
        loaded_container.headers = dump_container.headers
        if isinstance(loaded_container, Survey):
            assert_surveys_equal(dump_container, loaded_container)
        else:
            assert_gathers_equal(dump_container, loaded_container)
