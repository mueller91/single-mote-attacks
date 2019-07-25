#!/usr/bin/env python
# coding: utf8

import numpy as np
import pandas
from scipy.interpolate import CubicSpline


class CubicSplineWrapper:
    def __init__(self, x, y, bandwidth, *args, **kwargs):
        # save important values
        self.x, self.y = x, y
        self.left_interval_end = np.min(x)
        self.right_interval_end = np.max(x)
        self.bandwidth = bandwidth

        # create model
        self.model = CubicSpline(x, y, *args, **kwargs)

        # gette4r
        self.c = self.model.c

    def score_samples(self, samples):
        """
        Score array x of samples, extrapolating outside the [min, max] interval
        :param samples:
        :return:
        """
        assert isinstance(samples, pandas.Series), "This Wrapper takes only Series as Input!"
        spline_log_density = self.model(samples)
        result = np.where(np.logical_and(self.left_interval_end <= samples, samples <= self.right_interval_end),
                          spline_log_density,
                          self._extrapolate(samples))
        return result

    def _extrapolate(self, y):
        """
        Extrapolate a point
        :param y:
        :return:
        """
        # - x^2 / 2 b^2 + s(c)
        c = [np.max(self.x) if y[k] > np.max(self.x) else np.min(self.x) for k in y.index]
        distance_to_boundary = np.min((y-c, np.full(y.shape, 64.)), axis=0)
        bandwidth = max(self.bandwidth, 1./32.)
        return - 1 / (2 * bandwidth * bandwidth) * np.square(distance_to_boundary) + self.model(c)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
