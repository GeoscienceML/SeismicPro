"""Stores global `SeismicPro` configuration and provides an interface to interact with it.

The following options are currently available:

enable_fast_pickling : bool, defaults to False
    If enabled, several objects such as `SeismicDataset` or `Survey` stop pickling their most memory-intensive
    attributes such as `index` and `headers`. Allows for huge time and memory savings when running pipelines with `mpc`
    prefetch.

All options can be accessed and updated via `config` variable available directly from the global `seismicpro`
namespace.

Examples
--------
Display the current config state:
>>> from seismicpro import config
>>> print(config)
{'enable_fast_pickling': False}

Update config option globally:
>>> config["enable_fast_pickling"] = True
>>> print(config)
{'enable_fast_pickling': True}

Reset an option to its default value:
>>> config.reset_options("enable_fast_pickling")
>>> print(config)
{'enable_fast_pickling': False}

Temporarily change given options within a context manager:
>>> with config.use_options(enable_fast_pickling=True):
>>>     print(config)
>>> print(config)
{'enable_fast_pickling': True}
{'enable_fast_pickling': False}
"""

from contextlib import contextmanager


class Config:
    """Store global `SeismicPro` configuration and provide an interface to interact with it."""

    def __init__(self):
        self.default_options = {
            "enable_fast_pickling": False,
        }
        self.options = self.default_options.copy()

    def __repr__(self):
        """String representation of the current config state."""
        return repr(self.options)

    def __getitem__(self, option):
        """Get current option value."""
        return self.options[option]

    def __setitem__(self, option, value):
        """Set a new option value."""
        self.options[option] = value

    def reset_options(self, *options):
        """Reset given options to their default values."""
        for option in options:
            if option in self.default_options:
                self.options[option] = self.default_options[option]
            else:
                _ = self.options.pop(option, None)

    @contextmanager
    def use_options(self, **options):
        """Enter a context manager that temporarily changes given options and reverts them back upon exit."""
        self.options.update(options)
        yield
        self.reset_options(*options.keys())


config = Config()
