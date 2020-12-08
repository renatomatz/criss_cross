"""
Define the Cleaner base class, inherrited classes and basic cleaning operations.
"""

from glob import glob

import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetype


class Cleaner:
    """Cleaner instances import data from some source and applies a set of cleaning functions to it.

    This is meant to automate cleaning on files that must be stored in a format that is hard to
    analyse, or for files that must be transformed into a specific format for further analysis.
    """

    def __init__(self, files, cleaning_ops=None, import_kwargs=None, init_check=False):
        """Initialize Cleaner object and set up cleaning operations.
        Args:
            files: file locator used in self.get_data() to find data to import.
            cleaning_ops: list of cleaning operations. Each element must be 
                either a callable or a tuple of format 
                (callable, (args), {kwargs}), where args is a tuple of 
                positional arguments and kwargs is a dictionary of keyword 
                arguments.
            import_kwargs: dictionary of arguments to import function. 
            init_check: whether to check if import is working by calling
                self.get_data() upon initialization.
        """

        self.files = files
        if init_check:
            self.get_data()
        
        if cleaning_ops is None:
            cleaning_ops = list()
        elif not isinstance(cleaning_ops, list):
            cleaning_ops = [cleaning_ops]

        self.cleaning_ops = list()
        for op in cleaning_ops:

            args = list()
            kwargs = dict()

            func = op
            if isinstance(op, tuple):
                op_it = iter(op)
                func = next(op_it)

                for elem in op_it:

                    if isinstance(elem, (list, tuple)):
                        if len(args) != 0:
                            raise ValueError("only one argument list can be passed per operation")
                        args = elem
                    elif isinstance(elem, dict):
                        if len(kwargs) != 0:
                            raise ValueError("only one keyword argument dict can be passed per operation")
                        kwargs = elem
                    else:
                        raise TypeError("Element is of incorrect type")

            self.add_op(func, *args, **kwargs)

        if (import_kwargs is not None) and (not isinstance(import_kwargs, dict)):
            raise TypeError("argument import_kwargs must be a dict of keyword arguments to pd.read_csv")

        self.import_kwargs = import_kwargs if import_kwargs is not None else dict()

    def add_op(self, op, *args, **kwargs):
        """Add perform checks and add operation into the cleaning operation list.

        Args:
            op: Operation to be added. Must be callable.
            *args: Positional arguments to this operation.
            **kwargs: Keyword arguments to this operation.
        """

        if not callable(op):
            raise TypeError("opt must be callable")

        self.cleaning_ops.append((op, args, kwargs))

    def __call__(self):
        """Get data and apply cleaning operations
        """

        df = self.get_data()

        for op, op_args, op_kwargs in self.cleaning_ops:
            df = op(df, *op_args, **op_kwargs)

        return df

    def get_data(self):
        """Get data from source
        """
        raise NotImplementedError()


class FileCleaner(Cleaner):

    def __init__(self, files, cleaning_ops=None, import_kwargs=None, init_check=False, csv_only=False):

        self.csv_only = csv_only
        super().__init__(files, cleaning_ops=cleaning_ops, import_kwargs=import_kwargs, init_check=init_check)

    def get_data(self):
        """Get data from system files using a globbed file name
        """

        globbed = glob(self.files)

        csv_files = [f_name for f_name in globbed if f_name[-3:] == "csv"]

        if len(csv_only) == 0:
            raise IOError("No valid files were retrieved from location")

        if self.csv_only and (len(globbed) != len(csv_files)):
            raise IOError("Some globbed files do not have a .csv extention")

        return pd.concat([pd.read_csv(name, **self.import_kwargs) for name in csv_files])
