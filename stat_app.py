"""
Define StatApp base class and inheritted classes.
"""


from functools import partial
from collections import namedtuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

from sklearn.linear_model import LinearRegression

from entity import Entity
from ops import *


class StatApp:

    def __init__(self, entity, stat, cols=None, start=None, end=None):

        assert isinstance(entity, Entity)

        self.reset_all()

        self._data = None

        self._entity = entity
        self._stat = stat

        self.cols = cols if cols is not None else slice(None)

        self.start = start
        self.end = end

    @property
    def data(self):
        return self._full_data.sort_index().loc[self.start:self.end, self.cols]

    @property
    def _full_data(self):
        return self._entity.get_data(self._stat)

    def plot(self, ax=None, *args, **kwargs):

        ret, ax = self._make_plot_ret(ax)
        self.data.plot(ax=ax, *args, **kwargs)

        return ret

    @staticmethod
    def _make_plot_ret(ax, nrows=1, ncols=1, figsize=(12, 8)):
        ret = None
        if ax is None:
            ret = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
            ax = ret[1]
        elif (isinstance(ax, tuple) and len(ax) != nrows*ncols) or (nrows*ncols != 1):
            raise ValueError(f"Incorrect number of axes, there should be {nrows*ncols} passed onto the ax parameter")

        return ret, ax

    def reset_all(self):
        pass


class Simulator(StatApp):

    sim_info = namedtuple("sim_info", ["kind", "n_steps", "extra"])

    def __init__(self, entity, stat, start=None, end=None, cols=None,
                 sim_start=None,
                 past_events=None, 
                 fit_method="events",
                 lazy=False,
                 time_step="D"):

        super().__init__(entity, stat, start=start, end=end, cols=cols)

        if past_events is not None:
            self.fit_dates(past_events)

        self.sim_start = sim_start if sim_start is not None else end

        if not lazy and len(self.data.loc[sim_start].shape) > 1:
            raise ValueError("shock date returns more than one value on data")

        if time_step[-1] not in list("DWMQY"):
            raise ValueError("time step must be one of D, W, M, Q, Y")

        self.time_step = time_step

    @property
    def data_ext(self):

        data = self.data.loc[:self.shock_date]

        if self._sim is None:
            return data

        return pd.concat([data, self._sim]) 

    @property
    def time_step(self):
        return self._time_step

    @time_step.setter
    def time_step(self, step):
        if step == "Q":
            self._time_step = np.timedelta64(4, 'M')
        else:
            try:
                self._time_step = np.timedelta64(1, step)
            except TypeError:
                raise TypeError("time step must be one of D, W, M, Q, Y")

    def plot_bar(self, ax=None):

        ret, ax = self._make_plot_ret(ax)

        df = pd.DataFrame(self.data.dropna())

        if len(df.columns == 1):

            # If there is only one column, then color code the simulation
            # idk how to do that for multiplt columns

            df["color"] = "C0"

            if self._sim is not None:
                df1 = pd.DataFrame(self._sim[1:].dropna(), columns=df.columns[:-1])
                df1["color"] = "purple"
                df = pd.concat([df, df1])

            df.iloc[:,0].plot.bar(ax=ax, color=df["color"])

        else:
            df.plot.bar(ax=ax)

        ax.legend()

        return ret

    def plot(self, ax=None):

        ret, ax = self._make_plot_ret(ax)
        self.data.dropna().plot(ax=ax)

        if self._sim is not None:
            self._sim.dropna().plot(ax=ax, color="purple", label="Shock")

        ax.legend()

        return ret

    def fit(self, shock, method="shocks"):

        weight = 1 
        if method == "shocks":
            weight = 1 / (self._fitted_shocks + 1)
        elif method == "days":
            days = len(shock)
            weight = days / (self._fitted_days + days)
        elif method == "reset":
            self.reset()
        
        self._mean = (1-weight)*self._mean + weight*np.mean(shock)
        self._var = (1-weight)*self._var + weight*np.var(shock)

        self.reset_sim()

    def fit_dates(self, past_shock, method="shocks"):

        data = self._full_data.pct_change()

        if isinstance(past_shock, list):
            for start, end in past_shock:
                self.fit(data.loc[start:end], method=method)
        elif isinstance(past_shock, tuple):
            start, end = past_shock
            self.fit(data.loc[start:end], method=method)
        else:
            raise TypeError("past_shocks must be either a tuple or a list of tuples")

    def reset_fit(self):
        self._fitted_shocks = 0
        self._fitted_days = 0

        self._mean = 0
        self._var = 0.0005 # remember this is for pct changes

    def set_sim(self, sim_vals):

        stepper, self._sim = Stepper.make_sim_stepper(self.data.loc[:self.shock_date], self.time_step)

        for val in sim_vals:
            stepper(val)

        self.last_sim = Simulator.sim_info(kind="set", n_steps=len(sim_vals), extra=None)

    def simple_sim(self, n_steps):

        stepper, self._sim = Stepper.make_sim_stepper(self.data.loc[:self.shock_date], self.time_step)

        for _ in range(n_steps+1):
            stepper(self._sim.iloc[-1] * (1+np.random.normal(self._mean, np.sqrt(self._var))))

        self.last_sim = Simulator.sim_info(kind="simple", n_steps=n_steps, extra=None)

    def mean_target_sim(self, n_steps, mean, var=None, sub_min=False):

        var = var if var is not None else self._var

        data = self.data

        if isinstance(mean, dict):
            try:
                mean = [mean[col] for col in data.columns]
            except KeyError:
                raise ValueError("If mean is a dict, it must have at least the datas columns")

        if isinstance(mean, list):
            mean = np.array(mean)

        req_mean = mean / n_steps

        stepper, sim = Stepper.make_sim_stepper(data.loc[:self.shock_date], self.time_step)

        # there are problems with simulating change when values become negative
        # there is a bias towards zero or an exponential decrease towards negative infinity
        orig_min = 0
        if sub_min:
            orig_min = np.min(data) 
        sim -= orig_min

        for _ in range(n_steps+1):
            last = sim.iloc[-1]
            change = (1+np.random.normal(req_mean, np.sqrt(var)))
            stepper(last * change)

        self.last_sim = Simulator.sim_info(kind="mean_target", n_steps=n_steps, extra=None)
        self._sim = sim + orig_min

    def dynamic_stat_sim(self, n_steps, mean_func=None, var_func=None):

        stepper, self._sim = Stepper.make_sim_stepper(self.data.loc[:self.shock_date], self.time_step)

        for i in range(n_steps+1):
            stepper(
                self._sim.iloc[-1] 
                * (1+np.random.normal(mean_func(self._mean, i), 
                                      np.sqrt(var_func(self._var, i))))
            )

        self.last_sim = Simulator.sim_info(kind="dynamic", n_steps=n_steps, extra=None)

    def get_sim(self):
        return self._sim

    def reset_sim(self):
        self.last_sim = None
        self._sim = None

    def make_parasite(self, f, *args, **kwargs):
        return Parasite(self, f, *args, **kwargs)

    def reset_all(self):
        super().reset_all()
        self.reset_fit()
        self.reset_sim()


class Stepper:

    def __init__(self, df, time_step):
        self._df = df
        self._time_step = time_step

    def __call__(self, new_val):
        self._df.loc[(self._df.index[-1]+self._time_step).normalize()] = new_val

    def get_data(self):
        return self._df
        
    @staticmethod
    def make_sim_stepper(df, time_step):
            
        first_val = None
        new_df = None
        if isinstance(df, pd.DataFrame):
            first_val = df.iloc[-1].values.copy() 
            new_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=df.columns.tolist(), dtype=float)
        elif isinstance(df, pd.Series):
            first_val = df.iloc[-1]
            new_df = pd.Series(index=pd.DatetimeIndex([]), dtype=float)

        # first value is the last value of input df
        new_df.loc[df.index[-1]] = first_val

        stepper = Stepper(new_df, time_step)
        return stepper, stepper.get_data()


class TSShock(Simulator):

    def __init__(self, entity, stat, start=None, end=None, cols=None,
                 sim_start=None, past_events=None, fit_method="events", lazy=True, time_step="D",
                 shock_date=None):

        super().__init__(entity, stat, start=start, end=end, cols=cols,
                         sim_start=sim_start, past_events=past_events, fit_method=fit_method, 
                         lazy=lazy, time_step=time_step)

        self.shock_date = shock_date if shock_date is not None else self.sim_start

        if not lazy:
            if not isinstance(self.data.index, pd.DatetimeIndex):
                raise TypeError("Data must be a time series")

    def plot_ma(self, windows, ax=None, name=""):

        ret, ax = self._make_plot_ret(ax)

        data = self.data_ext

        ma_dict = {w: data.rolling(w).mean() for w in windows}

        for w, data in ma_dict.items():
            ax.plot(data.index, data,
                    linestyle="--",
                    label=f"{name} {w} Day Moving Average")

        ax.legend()

        return ret

    def plot_agg(self, agg_func=np.mean, ax=None, name=""):

        ret, ax = self._make_plot_ret(ax)

        data = self.data_ext

        stat = data.apply(agg_func)
        
        ax.hlines(stat, min(data.index), max(data.index), linestyle="--", label=name)
        
        rng = data.min() - data.max()

        for _, n in stat.items():
            ax.text(min(data.index), stat+(rng/25), str(np.round(n, 2)))
        
        ax.legend() 

        return ret

    def plot_peak(self, include_trough=False, text=True, cross=False, text_pos=0.8, ax=None):

        ret, ax = self._make_plot_ret(ax)

        # This won't work with pd.Series

        data = self.data_ext
        peak_trough = self.get_peak_trough()

        for name in peak_trough.columns.get_level_values(0).unique().tolist():
            # iterates over the names on the top levels
            sub_df = peak_trough[name]
            events = ["Peak"] + ["Trough"]*include_trough
            colors = ["Green"] + ["Red"]*include_trough

            trough = False
            for event, color in zip(events, colors):
                val = sub_df.loc[event, "Value"]
                ax.axhline(val, color=color, alpha=0.5, label=name+" "+event) 
                if cross:
                    ax.axvline(sub_df.loc[event, "Date"], color=color, alpha=0.5)
                if text:
                    text = str(round(val, 2))
                    if event == "Trough":
                        # ew... what have you become...
                        pval = sub_df.loc["Peak", "Value"]
                        text += f" ({str(round((val-pval)/pval, 2))})"
                    self.plot_h_text(data[name], val, text, ax, text_up=trough, text_pos=text_pos)
                trough = True

        ax.legend()

        return ret
    
    @staticmethod
    def plot_h_text(data, h_val, text, ax, text_up=False, text_pos=.8):

        if not isinstance(text_pos, float) or not (0 <= text_pos <= 1):
            raise ValueError("Text pos must be a float between 0 and 1")

        shift = (np.max(data) - np.min(data))/25
        end = data.index[int(text_pos*len(data))]

        text_up = 1 if text_up else -1

        ax.text(end, h_val+text_up*shift, text)

    def get_peak_trough(self):

        data = self.data_ext

        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)

        cols = [(l1, l2) for l1 in data.columns for l2 in ["Date", "Value"]]

        ret = pd.DataFrame(index=["Peak", "Trough"],
                           columns = pd.MultiIndex.from_product([data.columns.tolist(), ["Date", "Value"]]))

        for name, col in data.items():

            ret.loc["Peak", (name, "Value")] = col.max()
            ret.loc["Peak", (name, "Date")] = col.index[col == ret.loc["Peak", (name, "Value")]][0]

            after_peak = col.loc[ret.loc["Peak", (name, "Date")]:]

            ret.loc["Trough", (name, "Value")] = after_peak.min()
            ret.loc["Trough", (name, "Date")] = after_peak.index[after_peak == ret.loc["Trough", (name, "Value")]] 

        return ret


    def plot_shock(self, ax=None, name=""):

        ret, ax = self._make_plot_ret(ax)

        ax.axvline(self.data.index[-1], color="purple", alpha=0.5, label="Shock") 

        ax.legend()

        return ret

    def plot_lm(self, shock_sep=False, ax=None, formula_label=True, name=""):

        ret, ax = self._make_plot_ret(ax)

        data = self.data_ext
        pivots = [self.shock_date] if shock_sep else None
        names = "formula" if formula_label else None

        self.multi_lm(data, ax, pivots=pivots, labels=names)

        ax.legend()

        return ret


    @staticmethod
    def multi_lm(data, ax, pivots=None, labels=None):

        # the original pivots parameter will be used later
        pivs = pivots if pivots is not None else []

        pivs = [None] + pivs + [None]

        if isinstance(data, pd.DataFrame):
            for label, content in data.items():
                TSShock.multi_lm(content, ax, pivots=pivots, labels=labels)

        elif isinstance(data, pd.Series):

            for i in range(len(pivs)-1):

                y = data[pivs[i]:pivs[i+1]]
                linspace = np.arange(len(y))

                lm = LinearRegression().fit(linspace.reshape(-1, 1), y)

                Y = lm.intercept_ + linspace*lm.coef_[0]

                label = None
                if labels == "formula":
                    label = f"y = {round(lm.coef_[0], 4)}x + {round(lm.intercept_, 2)}"

                ax.plot(y.index, Y, linestyle="--", alpha=0.5, label=label)

        else:
            raise TypeError(f"Invalid type {type(data)}")

class BondApp(TSShock):

    @staticmethod
    def get_term(df, term):

        if isinstance(term, str):

            if term not in ["short", "medium", "long"]:
                raise ValueError("Invalid term.")

            letter = "M" if term == "short" else "Y"
            mult = 10 if term == "long" else 1

            cols = []
            for col in df.columns:
                n, l = int(col[:-1]), col[-1]
                if (term == "short") and (l == "M" or n == 1):
                    cols.append(col)
                elif (term == "medium") and (l == "Y" and (1 < n < 10)):
                    cols.append(col)
                elif (term == "long") and (l == "Y" and n >= 10):
                    cols.append(col)

            return df[cols]

        elif isinstance(term, list):

            return pd.concat([get_term(df, t) for t in term], join="outer", axis=1)

    @staticmethod
    def plot_yield(data, ax, colors, norm, int_time="int_time", **kwargs):
        data.drop(int_time).plot(ax=ax,
                                 color=colors(norm(data[int_time])),
                                 **kwargs)

    def plot_yield_movement(self, major="Q", minor="M", ax=None, label_maj=True):

        ret, ax = self._make_plot_ret(ax)

        data = self.data_ext

        colors = cm.get_cmap("viridis", 20)

        data = data.rename(
            {**{f"{n}M": float(n/12) for n in range(1, 13)},
             **{f"{n}Y": float(n) for n in range(1, 31)}},
            axis=1
        )

        # This is key to color-coding date progression with the pallete
        data["int_time"] = data.index.astype(int)
        mn = data["int_time"].min()
        mx = data["int_time"].max()

        # function to normalize date to be within 0 and 1
        norm = lambda x: (x - mn)/(mx - mn)

        labels = []

        self.plot_yield(data.iloc[0], ax, colors, norm, linewidth=4)
        labels.append(data.iloc[0].name.strftime("%Y-%m-%d"))

        if major is not None:

            for idx, row in data.resample(major).last().iterrows():
                self.plot_yield(row, ax, colors, norm, linewidth=4)
                labels.append(idx.strftime("%Y-%m-%d"))

            labels.pop()
            labels.append(data.iloc[-1].name.strftime("%Y-%m-%d"))

        ax.legend(labels)

        if minor is not None:
            for idx, row in data.resample(minor).last().iterrows():
                self.plot_yield(row, ax, colors, norm, alpha=0.3)

        ax.set_xlabel("Years to Maturity")
        ax.set_ylabel("Yield (%)")

        return ret

    def plot_terms(self, plot_shock=True, ax=None):

        ret, ax = self._make_plot_ret(ax, nrows=3, ncols=1, figsize=(16, 20))

        data = self.data
        sim = self._sim if self._sim is not None else pd.DataFrame(columns=data.columns)

        for i, term in enumerate(["short", "medium", "long"]):

            term_ax = ax[i]

            self.get_term(data, term).plot(ax=term_ax)
            self.get_term(sim, term).plot(ax=term_ax, color="purple")

            term_ax.set_title(term+ " term")
            term_ax.legend()

            if plot_shock:
                ax[i].axvline(data.index[-1],
                              color="purple", 
                              alpha=0.5, 
                              label="Shock") 

        return ret

    def plot_agg_terms(self, ax=None):

        ret, ax = self._make_plot_ret(ax)

        data = self.data_ext

        for i, term in enumerate(["short", "medium", "long"]):
            ax.plot(self.get_term(data, term).mean(axis=1), label=term+" term")

        ax.legend()

        return ret


class Parasite(StatApp):

    def __init__(self, app, f, *args, **kwargs):

        app.reset_all()
        self.app = app
        self.f = partial(f, *args, **kwargs)

    @property
    def data(self):
        return self.app.data.apply(self.f)

    @property
    def _full_data(self):
        return self.app._full_data.apply(self.f)

    def __getattribute__(self, name):
        """Now I am become Death, the destroyer of worlds"""

        if name in ["app", "f", "data", "_full_data"]:
            return super().__getattribute__(name)

        return self.app.__getattribute__(name)

    def __setattr__(self, name, value):

        if name in ["app", "f"]:
            return super().__setattr__(name, value)
        else:
            raise AttributeError("Can't set attributes through Parasite")
