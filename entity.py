"""
Define Entitity instances
"""

import numpy as np
import pandas as pd

from cleaner import FileCleaner


class Entity:
    """Entity instances are essentially data managers that get data from a 
    a source and, if specified, caches it. This caching mechanism is 
    implemented in various inherited classes.
    """

    def __init__(self, cleaners=None, data=None, cache=True, lazy=True, default_cleaner=FileCleaner):
        """Initialize Entity and set up data cleaners.

        Args:
            cleaners: dictionary of cleaners with a {name: cleaner} format.
                the "cleaner" can be just a file locator that is passed as the
                first argument to a the default_cleaner type.
            data: any local data yout want to be part of this entity. Use
                the same format as the cleaners.
            cache: whether to cache data after being retrieved.
            lazy: whether to cache all data upon initialization or upon
                first request. 
            default_cleaner: if a file locator is specified in the cleaners
                parameter, which cleaner should be used to fetch its information.
        """

        assert isinstance(data, dict) or data is None
        assert isinstance(cleaners, dict) or cleaners is None

        self.cleaners = dict()

        self.cleaners = cleaners if cleaners is not None else dict()

        if not isinstance(default_cleaner, type):
            raise TypeError("default_cleaner must be of type <type>")

        if lazy:
            for name, cleaner in self.cleaners.items():
                if not isinstance(cleaner, Cleaner):
                    self.cleaners[name] = Cleaner(cleaner, csv_check=True)
        else:
            for name, cleaner in self.cleaners.items():
                if isinstance(cleaner, Cleaner):
                    self.data[name] = cleaner()
                else:
                    self.data[name] = default_cleaner(cleaner)()

        self.data = data if data is not None else dict()

        self.cache = cache

    def get_data(self, name):
        """Get data from source.

        Args:
            name: name of the data (specified upon initialization)
        """
        raise NotImplementedError()


class LocalEntity:
    """Caches data locally"""
        
    def get_data(self, name):
        
        if name in self.data:
            return self.data[name]

        elif name in self.cleaners:
            if self.cache:
                self.data[name] = self.cleaners[name]()
                return self.data[name]
            return self.cleaners[name]()

        else:
            raise ValueError(f"{name} is not a valid statistic")
