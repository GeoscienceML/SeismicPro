"""Implements a MuterField class which stores muters defined at different field locations and allows for their spatial
interpolation"""

from .muter import Muter
from ..field import ValuesAgnosticField, VFUNCFieldMixin


class MuterField(ValuesAgnosticField, VFUNCFieldMixin):
    """A class for storing and interpolating muters over a field.

    A field can be populated with muters in 4 main ways:
    - by passing precalculated muters in the `__init__`,
    - by creating an empty field and then iteratively updating it with calculated muters using `update`,
    - by loading a field from a file of vertical functions via its `from_file` `classmethod`,
    - by constructing a muter field to attenuate high amplitudes immediately following the first breaks from a
      near-surface velocity model via its `from_refractor_velocity_field` `classmethod`.

    After all muters are added, field interpolator should be created to make the field callable. It can be done either
    manually by executing `create_interpolator` method or automatically during the first call to the field if
    `auto_create_interpolator` flag was set to `True` upon field instantiation. Manual interpolator creation is useful
    when one wants to fine-tune its parameters or the field should be later passed to different processes (e.g. in a
    pipeline with prefetch with `mpc` target) since otherwise the interpolator will be independently created in all the
    processes.

    Examples
    --------
    A field can be created empty and updated with instances of `Muter` class:
    >>> field = MuterField()
    >>> muter = Muter(offsets=[0, 1000, 2000, 3000], times=[100, 300, 500, 700],
    ...               coords=Coordinates((50, 50), names=("INLINE_3D", "CROSSLINE_3D")))
    >>> field.update(muter)

    Or created from precalculated instances:
    >>> field = MuterField(list_of_muters)

    Or simply loaded from a file of vertical functions:
    >>> field = MuterField.from_file(path)

    Some task-specific muters can be created directly from fields of other types, e.g. a first-break muter can be
    created from a `RefractorVelocityField`:
    >>> field = MuterField.from_refractor_velocity_field(rvf, delay=50, velocity_reduction=150)

    Field interpolator will be created automatically upon the first call by default, but one may do it explicitly by
    executing `create_interpolator` method:
    >>> field.create_interpolator("idw")

    Now the field allows for muter interpolation at given coordinates:
    >>> muter = field((10, 10))

    Or can be passed directly to some gather processing methods:
    >>> gather = survey.sample_gather().mute(field)

    Parameters
    ----------
    items : Muter or list of Muter, optional
        Muters to be added to the field on instantiation. If not given, an empty field is created.
    survey : Survey, optional
        A survey described by the field.
    is_geographic : bool, optional
        Coordinate system of the field: either geographic (e.g. (CDP_X, CDP_Y)) or line-based (e.g. (INLINE_3D,
        CROSSLINE_3D)). Inferred automatically on the first update if not given.
    auto_create_interpolator : bool, optional, defaults to True
        Whether to automatically create default interpolator (IDW) upon the first call to the field.

    Attributes
    ----------
    survey : Survey or None
        A survey described by the field. `None` if not specified during instantiation.
    item_container : dict
        A mapping from coordinates of field items as 2-element tuples to the items themselves.
    is_geographic : bool
        Whether coordinate system of the field is geographic. `None` for an empty field if was not specified during
        instantiation.
    coords_cols : tuple with 2 elements or None
        Names of SEG-Y trace headers representing coordinates of items in the field if names are the same among all the
        items and match the geographic system of the field. ("X", "Y") for a field in geographic coordinate system if
        names of coordinates of its items are either mixed or line-based. ("INLINE_3D", "CROSSLINE_3D") for a field in
        line-based coordinate system if names of coordinates of its items are either mixed or geographic. `None` for an
        empty field.
    interpolator : SpatialInterpolator or None
        Field data interpolator.
    is_dirty_interpolator : bool
        Whether the field was updated after the interpolator was created.
    auto_create_interpolator : bool
        Whether to automatically create default interpolator (IDW) upon the first call to the field.
    """
    item_class = Muter

    def construct_item(self, items, weights, coords):
        """Construct a new muter by averaging other muters with corresponding weights.

        Parameters
        ----------
        items : list of Muter
            Muters to be aggregated.
        weights : list of float
            Weight of each item in `items`.
        coords : Coordinates
            Spatial coordinates of a muter being constructed.

        Returns
        -------
        item : Muter
            Constructed muter instance.
        """
        return self.item_class.from_muters(items, weights, coords=coords)

    @classmethod
    def from_refractor_velocity_field(cls, field, delay=0, velocity_reduction=0):
        """Create a muter field to attenuate high amplitudes immediately following the first breaks by constructing a
        muter for each velocity model in a `field` in the following way:
        1) Reduce velocity of each refractor by `velocity_reduction` to account for imperfect model fit and variability
           of first arrivals at a given offset,
        2) Take an offset-time curve defined by an adjusted model from the previous step and shift it by `delay` to
           handle near-offset traces.

        Parameters
        ----------
        field : RefractorVelocityField
            A near-surface velocity model to construct a muter field from.
        delay : float, optional, defaults to 0
            Introduced constant delay. Measured in milliseconds.
        velocity_reduction : float or array-like of float, optional, defaults to 0
            A value used to decrement velocity of each refractor. If a single `float`, the same value is used for all
            refractors. Measured in meters/seconds.

        Returns
        -------
        field : MuterField
            Created muter field.
        """
        items = [cls.item_class.from_refractor_velocity(item, delay=delay, velocity_reduction=velocity_reduction)
                 for item in field.item_container.values()]
        return cls(items, survey=field.survey, is_geographic=field.is_geographic,
                   auto_create_interpolator=field.auto_create_interpolator)
