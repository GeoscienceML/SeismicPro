"""Implements SeismicIndex class that allows for iteration over gathers in a survey or a group of surveys"""

import os
import warnings
from functools import wraps, reduce
from textwrap import indent, dedent

import numpy as np
import pandas as pd
from batchflow import DatasetIndex

from .survey import Survey
from .containers import GatherContainer
from .utils import to_list, maybe_copy
from .const import HDR_TRACE_POS


class IndexPart(GatherContainer):
    """A class that represents a part of `SeismicIndex` which contains trace headers of several surveys being merged
    together.

    Parameters
    ----------
    headers : pd.DataFrame
        Trace headers of surveys in the index part. Must have `MultiIndex` columns with two levels: the first one with
        names of surveys in the part and the second with trace headers of a particular survey.
    common_headers : set of str
        Trace headers with common values among all the surveys in the part (e.g. keys used to merge the surveys). Used
        to speed up merge/apply/filter operations.
    surveys_dict : dict
        A mapping from survey names from the first level of `headers.columns` to the surveys themselves.
    indexer : BaseIndexer, optional
        An indexer of `headers`. Created automatically if not given.
    copy_headers : bool, optional, defaults to False
        Whether to copy `headers` while constructing the part.
    """

    def __init__(self, headers, common_headers, surveys_dict, indexer=None, copy_headers=False):
        headers = headers.copy(copy_headers)
        if indexer is None:  # Force indexer creation on headers setting
            self.headers = headers
        else:  # Use existing indexer to speed up part creation
            self._headers = headers
            self._indexer = indexer
        self.common_headers = common_headers
        self.surveys_dict = surveys_dict

    @property
    def survey_names(self):
        """list of str: names of surveys in the index part."""
        return sorted(self.surveys_dict.keys())

    @classmethod
    def from_survey(cls, survey, copy_headers=False):
        """Construct an index part from a single survey."""
        if not isinstance(survey, Survey):
            raise ValueError("survey must be an instance of Survey")

        headers = survey.headers.copy(deep=False)
        common_headers = set(headers.columns)
        headers.columns = pd.MultiIndex.from_product([[survey.name], headers.columns])

        # pylint: disable-next=protected-access
        return cls(headers, common_headers, {survey.name: survey}, indexer=survey._indexer, copy_headers=copy_headers)

    @staticmethod
    def _filter_equal(headers, header_cols):
        """Keep only those rows of `headers` where values of given headers are equal in all surveys."""
        if not header_cols:
            return headers
        drop_mask = np.column_stack([np.ptp(headers.loc[:, (slice(None), col)], axis=1).astype(np.bool_)
                                     for col in header_cols])
        return headers.loc[~np.any(drop_mask, axis=1)]

    def merge(self, other, on=None, validate_unique=True, copy_headers=False):
        """Create a new `IndexPart` by merging trace headers of `self` and `other` on given common headers."""
        self_indexed_by = set(to_list(self.indexed_by))
        other_indexed_by = set(to_list(other.indexed_by))
        if self_indexed_by != other_indexed_by:
            raise ValueError("All parts must be indexed by the same headers")
        if set(self.survey_names) & set(other.survey_names):
            raise ValueError("Only surveys with unique names can be merged")

        possibly_common_headers = self.common_headers & other.common_headers
        if on is None:
            on = possibly_common_headers - {"TRACE_SEQUENCE_FILE", HDR_TRACE_POS}
            left_df = self.headers
            right_df = other.headers
        else:
            on = set(to_list(on)) - self_indexed_by
            # Filter both self and other by equal values of on
            left_df = self._filter_equal(self.headers, on - self.common_headers)
            right_df = self._filter_equal(other.headers, on - other.common_headers)
        headers_to_check = possibly_common_headers - on

        merge_on = sorted(on)
        left_survey_name = self.survey_names[0]
        right_survey_name = other.survey_names[0]
        left_on = to_list(self.indexed_by) + [(left_survey_name, header) for header in merge_on]
        right_on = to_list(other.indexed_by) + [(right_survey_name, header) for header in merge_on]

        validate = "1:1" if validate_unique else "m:m"
        headers = pd.merge(left_df, right_df, how="inner", left_on=left_on, right_on=right_on, copy=copy_headers,
                           sort=False, validate=validate)

        # Recalculate common headers in the merged DataFrame
        common_headers = on | {header for header in headers_to_check
                                      if headers[left_survey_name, header].equals(headers[right_survey_name, header])}
        return type(self)(headers, common_headers, {**self.surveys_dict, **other.surveys_dict})

    def create_subset(self, indices):
        """Return a new `IndexPart` based on a subset of its indices given."""
        subset_headers = self.get_headers_by_indices(indices)
        return type(self)(subset_headers, self.common_headers, self.surveys_dict)

    def copy(self, ignore=None):
        """Perform a deepcopy of all part attributes except for `surveys_dict`, `_indexer` and those specified in
        `ignore`, which are kept unchanged."""
        ignore = set() if ignore is None else set(to_list(ignore))
        return super().copy(ignore | {"surveys_dict"})

    @wraps(GatherContainer.reindex)
    def reindex(self, new_index, inplace=False):
        old_index = to_list(self.indexed_by)
        new_index = to_list(new_index)
        new_index_diff = set(new_index) - set(old_index)
        old_index_diff = set(old_index) - set(new_index)
        if new_index_diff - self.common_headers:
            raise ValueError("IndexPart can be reindexed only with common headers")

        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        new_diff_list = list(new_index_diff)
        self.headers[new_diff_list] = self.headers[((self.survey_names[0], new_ix) for new_ix in new_diff_list)]
        super().reindex(new_index, inplace=True)

        # Copy old index to each survey
        for sur in self.survey_names:
            for old_ix in old_index_diff:
                self.headers[(sur, old_ix)] = self.headers[(old_ix, "")]

        # Drop unwanted headers
        cols_to_drop = ([(sur, new_ix) for sur in self.survey_names for new_ix in new_index_diff] +
                        [(old_ix, "") for old_ix in old_index_diff])
        self.headers.drop(columns=cols_to_drop, inplace=True)

        self.common_headers = (self.common_headers - new_index_diff) | old_index_diff
        return self

    @wraps(GatherContainer.filter)
    def filter(self, cond, cols, axis=None, unpack_args=False, inplace=False, **kwargs):
        cols = to_list(cols)
        survey_names = self.survey_names
        indexed_by = set(to_list(self.indexed_by))
        if (set(cols) - indexed_by) <= self.common_headers:
            # Filter only one survey since all of them share values of `cols` headers
            survey_names = [survey_names[0]]

        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        for sur in survey_names:
            sur_cols = [col if col in indexed_by else (sur, col) for col in cols]
            super().filter(cond, cols=sur_cols, axis=axis, unpack_args=unpack_args, inplace=True, **kwargs)
        return self

    @wraps(GatherContainer.apply)
    def apply(self, func, cols, res_cols=None, axis=None, unpack_args=False, inplace=False, **kwargs):
        cols = to_list(cols)
        res_cols = cols if res_cols is None else to_list(res_cols)

        survey_names = self.survey_names
        indexed_by = set(to_list(self.indexed_by))
        if (set(cols) - indexed_by) <= self.common_headers:
            # Apply func only to one survey since all of them share values of `cols` headers
            survey_names = [survey_names[0]]

        self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
        for sur in survey_names:
            sur_cols = [col if col in indexed_by else (sur, col) for col in cols]
            sur_res_cols = [(sur, col) for col in res_cols]
            super().apply(func, cols=sur_cols, res_cols=sur_res_cols, axis=axis, unpack_args=unpack_args, inplace=True,
                          **kwargs)

        # Duplicate results for all surveys if func was applied only to the first one
        if len(survey_names) == 1:
            for sur in self.survey_names[1:]:
                self[[(sur, col) for col in res_cols]] = self[[(survey_names[0], col) for col in res_cols]]
            self.common_headers |= set(res_cols)
        return self


def delegate_to_parts(*methods):
    """Implement given `methods` of `SeismicIndex` by calling the corresponding method of its parts. In addition to all
    the arguments of the method of a part each created method accepts `recursive` flag which defines whether to process
    `train`, `test` and `validation` subsets of the index in the same manner if they exist."""
    def decorator(cls):
        for method in methods:
            def method_fn(self, *args, method=method, recursive=True, inplace=False, **kwargs):
                self = maybe_copy(self, inplace)  # pylint: disable=self-cls-assignment
                for part in self.parts:
                    getattr(part, method)(*args, inplace=True, **kwargs)
                # Explicitly reset iter since index parts were modified
                self.reset("iter")

                if recursive:
                    for split in self.splits.values():
                        getattr(split, method)(*args, recursive=True, inplace=True, **kwargs)
                return self
            setattr(cls, method, method_fn)
        return cls
    return decorator


@delegate_to_parts("reindex", "filter", "apply")
class SeismicIndex(DatasetIndex):
    """A class that enumerates gathers in a survey or a group of surveys and allows iterating over them.

    While `Survey` describes a single SEG-Y file, `SeismicIndex` is primarily used to describe survey concatenation
    (e.g. when several fields are being processed in the same way one after another) or merging (e.g. when traces from
    the same field before and after a given processing stage must be matched and compared).

    `SeismicIndex` consists of parts - instances of `IndexPart` class stored in `parts` attribute. Parts act as an
    additional SEG-Y file identifier after concatenation since different surveys may have non-unique `indices` making
    it impossible to recover a source survey for a given gather by its index. Each part in turn represents several
    surveys being merged together. It contains the following main attributes:
    - `indices` - unique identifiers of gathers in the part,
    - `headers` - merged trace headers from underlying surveys,
    - `surveys_dict` - a mapping from a survey name to the survey itself to further load traces.

    Thus a gather in a `SeismicIndex` is identified by values of its `header_index`, part and survey name. It can be
    obtained by calling :func:`~SeismicIndex.get_gather`. Iteration over gathers in the index is generally performed
    via :func:`~SeismicIndex.next_batch`.

    A complete algorithm of index instantiation looks as follows:
    1. Independently transform each argument to `SeismicIndex`:
        - instance of `SeismicIndex` is kept as is,
        - `Survey` is transformed to a single part. Its `headers` replicate survey `headers` except for a new level
          added to `DataFrame` columns with the name of the survey. This is done to avoid headers collisions during
          subsequent merges.
       In both cases input `headers` can optionally be copied.
    2. If a single argument was processed on the previous step, an index is already created.
    3. Otherwise combine parts of created indices depending on the `mode` provided:
        - "c" or "concat": Parts of the resulting index is simply a concatenation of all input parts with preserved
          order. All parts must contain surveys with same `name`s.
        - "m" or "merge": Parts with same ordinal numbers are combined together by merging their `headers`. The number
          of parts in all inputs must match and all the underlying surveys must have different `name`s.
       In both cases all parts must be indexed by the same trace headers.

    Examples
    --------
    Let's consider 4 surveys describing a single field before and after processing. Note that all of them have the same
    `header_index`:
    >>> s1_before = Survey(path, header_index=index_headers, name="before")
    >>> s2_before = Survey(path, header_index=index_headers, name="before")

    >>> s1_after = Survey(path, header_index=index_headers, name="after")
    >>> s2_after = Survey(path, header_index=index_headers, name="after")

    An index can be created from a single survey in the following way:
    >>> index = SeismicIndex(s1_before)

    If `s1_before` and `s2_before` represent different parts of the same field, they can be concatenated into one index
    to iterate over the whole field and process it at once. Both surveys must have the same `name`:
    >>> index = SeismicIndex(s1_before, s2_before, mode="c")

    Gathers before and after given processing stage can be matched using merge operation. Both surveys must have
    different `name`s:
    >>> index = SeismicIndex(s1_before, s1_after, mode="m")

    Merge can follow concat and vice versa. A more complex case, covering both operations is demonstrated below:
    >>> index_before = SeismicIndex(s1_before, s2_before, mode="c")
    >>> index_after = SeismicIndex(s1_after, s2_after, mode="c")
    >>> index = SeismicIndex(index_before, index_after, mode="m")

    Parameters
    ----------
    args : tuple of Survey, IndexPart or SeismicIndex
        A sequence of surveys, indices or parts to construct an index.
    mode : {"c", "concat", "m", "merge", None}, optional, defaults to None
        A mode used to combine multiple `args` into a single index. If `None`, only one positional argument can be
        passed.
    copy_headers : bool, optional, defaults to False
        Whether to copy `DataFrame`s of trace headers while constructing index parts.
    kwargs : misc, optional
        Additional keyword arguments to :func:`~SeismicIndex.merge` if the corresponding mode was chosen.

    Attributes
    ----------
    parts : tuple of IndexPart
        Parts of the constructed index.
    """
    def __init__(self, *args, mode=None, copy_headers=False, **kwargs):  # pylint: disable=super-init-not-called
        self.parts = tuple()
        self.train = None
        self.test = None
        self.validation = None

        if args:
            index = self.build_index(*args, mode=mode, copy_headers=copy_headers, **kwargs)
            self.__dict__ = index.__dict__
        elif kwargs:
            raise ValueError("No kwargs must be passed if an empty index is being created")

        self._iter_params = None
        self.reset("iter")

    @property
    def index(self):
        """tuple of pd.Index: Unique identifiers of seismic gathers in each part of the index."""
        return tuple(part.indices for part in self.parts)

    @property
    def n_parts(self):
        """int: The number of parts in the index."""
        return len(self.parts)

    @property
    def n_gathers_by_part(self):
        """int: The number of gathers in each part of the index."""
        return [part.n_gathers for part in self.parts]

    @property
    def n_gathers(self):
        """int: The number of gathers in the index."""
        return sum(self.n_gathers_by_part)

    @property
    def n_traces_by_part(self):
        """int: The number of traces in each part of the index."""
        return [part.n_traces for part in self.parts]

    @property
    def n_traces(self):
        """int: The number of traces in the index."""
        return sum(self.n_traces_by_part)

    @property
    def indexed_by(self):
        """str or list of str or None: Names of header indices of each part. `None` for empty index."""
        if self.is_empty:
            return None
        return self.parts[0].indexed_by

    @property
    def survey_names(self):
        """list of str or None: Names of surveys in the index. `None` for empty index."""
        if self.is_empty:
            return None
        return self.parts[0].survey_names

    @property
    def is_empty(self):
        """bool: Whether the index is empty."""
        return self.n_parts == 0

    @property
    def splits(self):
        """dict: A mapping from a name of non-empty train/test/validation split to its `SeismicIndex`."""
        return {split_name: getattr(self, split_name) for split_name in ("train", "test", "validation")
                                                      if getattr(self, split_name) is not None}

    def __len__(self):
        """The number of gathers in the index."""
        return self.n_gathers

    def get_index_info(self, index_path="index", indent_size=0, split_delimiter=""):
        """Recursively fetch index description string from the index itself and all the nested subindices."""
        if self.is_empty:
            return "Empty index"

        fold = (np.array(self.n_traces_by_part) / np.array(self.n_gathers_by_part)).astype(np.int32)
        info_df = pd.DataFrame({"Traces": self.n_traces_by_part, "Gathers": self.n_gathers_by_part, "Fold": fold},
                               index=pd.RangeIndex(self.n_parts, name="Part"))
        for sur in self.survey_names:
            info_df[f"Survey {sur}"] = [os.path.basename(part.surveys_dict[sur].path) for part in self.parts]

        msg = f"""
        Indexed by:                {", ".join(to_list(self.indexed_by))}
        Number of traces:          {self.n_traces}
        Number of gathers:         {self.n_gathers}
        Mean gather fold:          {int(self.n_traces / self.n_gathers)}
        Is split:                  {self.is_split}

        Statistics of {index_path} parts:
        """
        msg = indent(dedent(msg) + info_df.to_string() + "\n", " " * indent_size)

        # Recursively fetch info about index splits
        for split_name, split in self.splits.items():
            msg += split_delimiter + "\n" + split.get_index_info(f"{index_path}.{split_name}", indent_size+4,
                                                                 split_delimiter=split_delimiter)
        return msg

    def __str__(self):
        """Print index metadata including information about its parts and underlying surveys."""
        delimiter_placeholder = "{delimiter}"
        msg = self.get_index_info(split_delimiter=delimiter_placeholder)
        for i, part in enumerate(self.parts):
            for sur in part.survey_names:
                msg += delimiter_placeholder + f"\n\nPart {i}, Survey {sur}\n\n" + str(part.surveys_dict[sur]) + "\n"
        delimiter = "_" * max(len(line) for line in msg.splitlines())
        return msg.strip().format(delimiter=delimiter)

    def info(self):
        """Print index metadata including information about its parts and underlying surveys."""
        print(self)

    #------------------------------------------------------------------------#
    #                         Index creation methods                         #
    #------------------------------------------------------------------------#

    @classmethod
    def build_index(cls, *args, mode=None, copy_headers=False, **kwargs):
        """Build an index from `args` as described in :class:`~SeismicIndex` docs."""
        # Create an empty index if no args are given
        if not args:
            return cls(**kwargs)

        # Select an appropriate builder by passed mode
        if mode is None and len(args) > 1:
            raise ValueError("mode must be specified if multiple positional arguments are given")
        builders_dict = {
            None: cls.from_index,
            "m": cls.merge,
            "merge": cls.merge,
            "c": cls.concat,
            "concat": cls.concat,
        }
        if mode not in builders_dict:
            raise ValueError(f"Unknown mode {mode}")

        # Convert all args to SeismicIndex and combine them into a single index
        indices = cls._args_to_indices(*args)
        return builders_dict[mode](*indices, copy_headers=copy_headers, **kwargs)

    @classmethod
    def _args_to_indices(cls, *args):
        """Independently convert each positional argument to a `SeismicIndex`."""
        indices = []
        for arg in args:
            if isinstance(arg, Survey):
                builder = cls.from_survey
            elif isinstance(arg, IndexPart):
                builder = cls.from_parts
            elif isinstance(arg, SeismicIndex):
                builder = cls.from_index
            else:
                raise ValueError(f"Unsupported type {type(arg)} to convert to index")
            indices.append(builder(arg, copy_headers=False))
        return indices

    @classmethod
    def from_parts(cls, *parts, copy_headers=False):
        """Construct an index from its parts.

        Parameters
        ----------
        parts : tuple of IndexPart
            Index parts to convert to an index.
        copy_headers : bool, optional, defaults to False
            Whether to copy `headers` of parts.

        Returns
        -------
        index : SeismicIndex
            Constructed index.
        """
        if not parts:
            return cls()

        if not all(isinstance(part, IndexPart) for part in parts):
            raise ValueError("All parts must be instances of IndexPart")

        survey_names = parts[0].survey_names
        if any(survey_names != part.survey_names for part in parts[1:]):
            raise ValueError("Only parts with the same survey names can be concatenated into one index")

        indexed_by = parts[0].indexed_by
        if any(indexed_by != part.indexed_by for part in parts[1:]):
            raise ValueError("All parts must be indexed by the same columns")

        if copy_headers:
            parts = tuple(part.copy() for part in parts)

        index = cls()
        index.parts = parts
        index.reset("iter")
        return index

    @classmethod
    def from_survey(cls, survey, copy_headers=False):
        """Construct an index from a single survey.

        Parameters
        ----------
        survey : Survey
            A survey used to build an index.
        copy_headers : bool, optional, defaults to False
            Whether to copy survey `headers`.

        Returns
        -------
        index : SeismicIndex
            Constructed index.
        """
        return cls.from_parts(IndexPart.from_survey(survey, copy_headers=copy_headers))

    @classmethod
    def from_index(cls, index, copy_headers=False):
        """Construct an index from an already created `SeismicIndex`. Leaves it unchanged if `copy_headers` is `False`,
        returns a copy otherwise.

        Parameters
        ----------
        index : SeismicIndex
            Input index.
        copy_headers : bool, optional, defaults to False
            Whether to copy the index.

        Returns
        -------
        index : SeismicIndex
            Constructed index.
        """
        if not isinstance(index, SeismicIndex):
            raise ValueError("index must be an instance of SeismicIndex")
        if copy_headers:
            return index.copy()
        return index

    @classmethod
    def concat(cls, *args, copy_headers=False):
        """Concatenate `args` into a single index.

        Each positional argument must be an instance of `Survey`, `IndexPart` or `SeismicIndex`. All of them must be
        indexed by the same headers. Underlying surveys of different arguments must have same `name`s.

        Notes
        -----
        A detailed description of index concatenation can be found in :class:`~SeismicIndex` docs.

        Parameters
        ----------
        args : tuple of Survey, IndexPart or SeismicIndex
            Inputs to be concatenated.
        copy_headers : bool, optional, defaults to False
            Whether to copy `headers` of `args`.

        Returns
        -------
        index : SeismicIndex
            Concatenated index.
        """
        indices = cls._args_to_indices(*args)
        parts = sum([ix.parts for ix in indices], tuple())
        return cls.from_parts(*parts, copy_headers=copy_headers)

    @classmethod
    def merge(cls, *args, on=None, validate_unique=True, copy_headers=False):
        """Merge `args` into a single index.

        Each positional argument must be an instance of `Survey`, `IndexPart` or `SeismicIndex`. All of them must be
        indexed by the same headers. Underlying surveys of different arguments must have different `name`s.

        Notes
        -----
        A detailed description of index merging can be found in :class:`~SeismicIndex` docs.

        Parameters
        ----------
        args : tuple of Survey, IndexPart or SeismicIndex
            Inputs to be merged.
        on : str or list of str, optional
            Headers to be used as join keys. If not given, all common headers are used except for `TRACE_SEQUENCE_FILE`
            unless it is used to index `args`.
        validate_unique : bool, optional, defaults to True
            Check if merge keys are unique in all input `args`.
        copy_headers : bool, optional, defaults to False
            Whether to copy `headers` of `args`.

        Returns
        -------
        index : SeismicIndex
            Merged index.
        """
        indices = cls._args_to_indices(*args)
        if len({ix.n_parts for ix in indices}) != 1:
            raise ValueError("All indices being merged must have the same number of parts")
        indices_parts = [ix.parts for ix in indices]
        merged_parts = [reduce(lambda x, y: x.merge(y, on, validate_unique, copy_headers), parts)
                        for parts in zip(*indices_parts)]

        # Warn if the whole index or some of its parts are empty
        empty_parts = [i for i, part in enumerate(merged_parts) if not part]
        if len(empty_parts) == len(merged_parts):
            warnings.warn("Empty index after merge", RuntimeWarning)
        elif empty_parts:
            warnings.warn(f"Empty parts {empty_parts} after merge", RuntimeWarning)

        return cls.from_parts(*merged_parts, copy_headers=False)

    #------------------------------------------------------------------------#
    #                 DatasetIndex interface implementation                  #
    #------------------------------------------------------------------------#

    def index_by_pos(self, pos):
        """Return gather index and part by its position in the index.

        Parameters
        ----------
        pos : int
            Ordinal number of the gather in the index.

        Returns
        -------
        index : int or tuple
            Gather index.
        part : int
            Index part to get the gather from.
        """
        part_pos_borders = np.cumsum([0] + self.n_gathers_by_part)
        part = np.searchsorted(part_pos_borders[1:], pos, side="right")
        return self.indices[part][pos - part_pos_borders[part]], part

    def subset_by_pos(self, pos):
        """Return a subset of gather indices by their positions in the index.

        Parameters
        ----------
        pos : int or array-like of int
            Ordinal numbers of gathers in the index.

        Returns
        -------
        indices : list of pd.Index
            Gather indices of the subset by each index part.
        """
        pos = np.sort(np.atleast_1d(pos))
        part_pos_borders = np.cumsum([0] + self.n_gathers_by_part)
        pos_by_part = np.split(pos, np.searchsorted(pos, part_pos_borders[1:]))
        part_indices = [part_pos - part_start for part_pos, part_start in zip(pos_by_part, part_pos_borders[:-1])]
        return tuple(index[subset] for index, subset in zip(self.index, part_indices))

    def create_subset(self, index):
        """Return a new index object based on a subset of its indices given.

        Parameters
        ----------
        index : SeismicIndex or tuple of pd.Index
            Gather indices of the subset to create a new `SeismicIndex` object for. If `tuple` of `pd.Index`, each item
            defines gather indices of the corresponding part in `self`.

        Returns
        -------
        subset : SeismicIndex
            A subset of the index.
        """
        if isinstance(index, SeismicIndex):
            index = index.index
        if len(index) != self.n_parts:
            raise ValueError("Index length must match the number of parts")
        return self.from_parts(*[part.create_subset(ix) for part, ix in zip(self.parts, index)], copy_headers=False)

    #------------------------------------------------------------------------#
    #                     Statistics computation methods                     #
    #------------------------------------------------------------------------#

    def collect_stats(self, n_quantile_traces=100000, quantile_precision=2, limits=None, bar=True):
        """Collect the following trace data statistics for each survey in the index or a dataset:
        1. Min and max amplitude,
        2. Mean amplitude and trace standard deviation,
        3. Approximation of trace data quantiles with given precision.

        Since fair quantile calculation requires simultaneous loading of all traces from the file we avoid such memory
        overhead by calculating approximate quantiles for a small subset of `n_quantile_traces` traces selected
        randomly. Only a set of quantiles defined by `quantile_precision` is calculated, the rest of them are linearly
        interpolated by the collected ones.

        After the method is executed all calculated values can be obtained via corresponding attributes of the surveys
        in the index and their `has_stats` flag is set to `True`.

        Examples
        --------
        Statistics calculation for the whole index can be done as follows:
        >>> survey = Survey(path, header_index="FieldRecord", header_cols=["TraceNumber", "offset"], name="survey")
        >>> index = SeismicIndex(survey).collect_stats()

        Statistics can be calculated for a dataset as well:
        >>> dataset = SeismicDataset(index).collect_stats()

        After a train-test split is performed, `train` and `test` refer to the very same `Survey` instances. This
        allows for `collect_stats` to be used to calculate statistics for the training set and then use them to
        normalize gathers from the testing set to avoid data leakage during machine learning model training:
        >>> dataset.split()
        >>> dataset.train.collect_stats()
        >>> dataset.test.next_batch(1).load(src="survey").scale_standard(src="survey", use_global=True)

        Note that if no gathers from a particular survey were included in the training set its stats won't be
        collected!

        Parameters
        ----------
        n_quantile_traces : positive int, optional, defaults to 100000
            The number of traces to use for quantiles estimation.
        quantile_precision : positive int, optional, defaults to 2
            Calculate an approximate quantile for each q with `quantile_precision` decimal places. All other quantiles
            will be linearly interpolated on request.
        limits : int or tuple or slice, optional
            Time limits to be used for statistics calculation. `int` or `tuple` are used as arguments to init a `slice`
            object. If not given, `limits` passed to `Survey.__init__` are used. Measured in samples.
        bar : bool, optional, defaults to True
            Whether to show a progress bar.

        Returns
        -------
        self : same type as self
            An index or a dataset with collected stats. Sets `has_stats` flag to `True` and updates statistics
            attributes inplace for each of the underlying surveys.
        """
        for part in self.parts:
            for sur in part.surveys_dict.values():
                sur.collect_stats(indices=part.indices, n_quantile_traces=n_quantile_traces,
                                  quantile_precision=quantile_precision, limits=limits, bar=bar)
        return self

    #------------------------------------------------------------------------#
    #                            Loading methods                             #
    #------------------------------------------------------------------------#

    def get_gather(self, index, part=None, survey_name=None, limits=None, copy_headers=False, chunk_size=None,
                   n_workers=None):
        """Load a gather with given `index`.

        Parameters
        ----------
        index : int or 1d array-like
            An index of the gather to load. Must be one of `self.indices`.
        part : int
            Index part to get the gather from. May be omitted if index concatenation was not performed.
        survey_name : str or list of str
            Survey name to get the gather from. If several names are given, a list of gathers from corresponding
            surveys is returned. May be omitted if index merging was not performed.
        limits : int or tuple or slice or None, optional
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, `limits` passed to the corresponding `Survey.__init__` are used. Measured in samples.
        copy_headers : bool, optional, defaults to False
            Whether to copy the subset of index `headers` describing the gather.
        chunk_size : int, optional
            The number of traces to load by each of spawned threads. Loads all traces in the main thread by default.
        n_workers : int, optional
            The maximum number of simultaneously spawned threads to load traces. Defaults to the number of cpu cores.

        Returns
        -------
        gather : Gather or list of Gather
            Loaded gather instance. List of gathers is returned if several survey names was passed.
        """
        if part is None and self.n_parts > 1:
            raise ValueError("part must be specified if the index is constructed by concatenation")
        if part is None:
            part = 0
        index_part = self.parts[part]

        if survey_name is None and len(self.survey_names) > 1:
            raise ValueError("survey_name must be specified if the index is constructed by merging")
        if survey_name is None:
            survey_name = self.survey_names[0]

        is_single_survey = isinstance(survey_name, str)
        survey_names = to_list(survey_name)
        surveys = [index_part.surveys_dict[name] for name in survey_names]

        index_headers = index_part.get_headers_by_indices((index,))
        empty_headers = index_headers[[]]  # Handle the case when no headers were loaded for a survey
        gather_headers = [index_headers.get(name, empty_headers) for name in survey_names]

        gathers = [survey.load_gather(headers=headers, limits=limits, copy_headers=copy_headers,
                                      chunk_size=chunk_size, n_workers=n_workers)
                   for survey, headers in zip(surveys, gather_headers)]
        if is_single_survey:
            return gathers[0]
        return gathers

    def sample_gather(self, part=None, survey_name=None, limits=None, copy_headers=False, chunk_size=None,
                      n_workers=None):
        """Load a random gather from the index.

        Parameters
        ----------
        part : int
            Index part to sample the gather from. Chosen randomly if not given.
        survey_name : str
            Survey name to sample the gather from. If several names are given, a list of gathers from corresponding
            surveys is returned. Chosen randomly if not given.
        limits : int or tuple or slice or None, optional
            Time range for trace loading. `int` or `tuple` are used as arguments to init a `slice` object. If not
            given, `limits` passed to the corresponding `Survey.__init__` are used. Measured in samples.
        copy_headers : bool, optional, defaults to False
            Whether to copy the subset of index `headers` describing the gather.
        chunk_size : int, optional
            The number of traces to load by each of spawned threads. Loads all traces in the main thread by default.
        n_workers : int, optional
            The maximum number of simultaneously spawned threads to load traces. Defaults to the number of cpu cores.

        Returns
        -------
        gather : Gather or list of Gather
            Loaded gather instance. List of gathers is returned if several survey names was passed.
        """
        if part is None:
            part_weights = np.array(self.n_gathers_by_part) / self.n_gathers
            part = np.random.choice(self.n_parts, p=part_weights)
        if survey_name is None:
            survey_name = np.random.choice(self.survey_names)
        index = np.random.choice(self.parts[part].indices)
        return self.get_gather(index, part, survey_name, limits=limits, copy_headers=copy_headers,
                               chunk_size=chunk_size, n_workers=n_workers)

    #------------------------------------------------------------------------#
    #                       Index manipulation methods                       #
    #------------------------------------------------------------------------#

    def copy(self, ignore=None):
        """Perform a deepcopy of the index by copying its parts. All attributes of each part are deepcopied except for
        indexer, underlying surveys and those specified in `ignore`, which are kept unchanged.

        Parameters
        ----------
        ignore : str or array of str, defaults to None
            Part attributes that won't be copied.

        Returns
        -------
        copy : SeismicIndex
            Copy of the index.
        """
        parts_copy = [part.copy(ignore=ignore) for part in self.parts]
        self_copy = self.from_parts(*parts_copy, copy_headers=False)
        for split_name, split in self.splits.items():
            setattr(self_copy, split_name, split.copy())
        return self_copy
