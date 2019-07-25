import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from pretraining.classifier.base_classifier import BaseClassifier
from pretraining.classifier.cubic_spline_wrapper import CubicSplineWrapper


class KernelDensityClassifier(BaseClassifier):
    """
    PCA & Kernel density estimator classifier
    """

    def __init__(self, log, pca_components, atol, rtol, features_to_select=None, independand_bandwidth=False,
                 use_spline_fitting=False):
        self.spline = {}
        assert log is not None
        self.kde = None
        self.pca_components = pca_components
        self.atol = atol
        self.rtol = rtol
        self.independend_bandwidth = independand_bandwidth
        self.use_spline_fitting = use_spline_fitting
        BaseClassifier.__init__(self, logger=log, features_to_select=features_to_select, n_dim_reduction=pca_components)

    def get_model(self):
        """
        Getter for underlying model
        :return:            {'model name': model}
        """
        return {"kde": self.kde, "spline": self.spline}

    def load_pretrained_model(self, classifier_path):
        """
        Load a pretrained model from file and stores it as its kde/spline classifier.
        :param classifier_path:             string, Pickle file containing the classifier
        :return:                            None
        """
        with open(classifier_path, 'rb') as f:
            dict = pickle.load(f, encoding='utf8')
            self.kde = dict["kde"]
            self.spline = dict["spline"]

    def __str__(self):
        return "KernelDensityClassifier(pca_dim={}{})".format(
            self.pca_components, ", spline_fitted" if self.use_spline_fitting else "")

    def train_on_csv(self, training_data_file_csv, settings):
        """
        Wrapper for train_on_data
        :param training_data_file_csv:          str, Path to file containing training data.
        :param settings:                        batch config object, Configuration object
        :return:                                None
        """
        # read data
        self.logger.info('Kernel Density Classifier:\tFitting to\t\t' + training_data_file_csv)

        # create and return DataFrame
        df = pd.read_csv(training_data_file_csv, sep='\t')
        return self.train_on_data(df, settings)

    def train_on_data(self, df, settings):
        """
        Train classifier on some data.
        :param df:                  DataFrame, Contains data
        :param settings:            batch config object, Configuration object
        :return:                    None
        """

        # sanetize
        assert not df.empty
        df = self.sanetize_select(df, settings=settings)

        processed_data_df = self.fit_transform_data(df, settings)

        # use grid search cross-validation to optimize the bandwidth
        self.logger.debug('Kernel Density Classifier: \tPerforming Cross-Validation for optimal bandwith h ...')
        params = {'bandwidth': np.logspace(-1, 4, num=25)}
        # dependend kde bandwidth: sklearn uses a diagonal bandwidth matrix with a_ii = a_jj
        if not self.independend_bandwidth:
            grid = GridSearchCV(KernelDensity(atol=self.atol, rtol=self.rtol), params)
            # grid = RandomizedSearchCV(KernelDensity(), params)
            grid.fit(processed_data_df.values)
            self.kde = grid.best_estimator_
            if self.use_spline_fitting:
                raise NotImplementedError("Spline Fitting is only implemented for independant kernels!")
        # independent kde bandwidth: not supported by sklearn, thus we implement ourselves
        else:
            self.kde = {}
            for feat in processed_data_df.columns:
                # TRAIN KDE
                grid = GridSearchCV(KernelDensity(atol=self.atol, rtol=self.rtol), params)
                grid.fit(processed_data_df[feat].values.reshape(-1, 1))
                self.kde[feat] = grid.best_estimator_

                # FIT SPLINE
                x = np.linspace(start=0, stop=(processed_data_df[feat].max() + 1e-6) * 1.0, num=settings.j_spline_intervals)
                y = self.kde[feat].score_samples(x.reshape(-1, 1))    # no exp(), we want to copy KDE as close as posbl
                self.spline[feat] = CubicSplineWrapper(x, y, grid.best_estimator_.bandwidth)

    def classify(self, data_to_classify_csv, settings, truncate=None):
        """
        Wrapper for classify_on_data
        :param settings:
        :param data_to_classify_csv:    Path to csv file containing the data
        :param truncate:                If not none, truncate df by (a, b).
        :return:    Log density of samples
        """
        self.logger.info('Kernel Density Classifier: \tClassifying\t\t' + data_to_classify_csv)
        df = pd.read_csv(data_to_classify_csv, sep='\t')
        return self.classify_on_data(df=df,
                                     truncate=truncate, settings=settings)

    def classify_on_data(self, df, settings, truncate=None):
        """
        Classify data
        :param settings:
        :param df:        DataFrame containing the data
        :param truncate:                If not none, truncate df by (a, b).
        :return:    Log density of samples
        """
        # truncate for delta calculation
        if truncate is not None:
            self.logger.info('Kernel Density Classifier: \tTruncating to ' + str(truncate))
            df = df[truncate[0]: truncate[1]]
        # truncate in delta is another truncate: we cut of the startup phase
        df = self.sanetize_select(df, settings=settings)

        # apply normalisation, scaling, etc if set
        processed_data_df = self.transform_data(df, settings)

        # assert settings.features is superset of self.kde.keys()
        for k in settings.features:
            assert k in self.kde.keys(), "Using a feature which the KDE has not been pre trained on: {}".format(k)

        if not self.independend_bandwidth and not self.use_spline_fitting:
            out_kde = self.kde.score_samples(processed_data_df.values)
            out_spline = None
        elif not self.independend_bandwidth and self.use_spline_fitting:
            raise NotImplementedError("Not implemented!")
        elif self.independend_bandwidth:
            out_spline = 0.0
            out_kde = 0.0
            for feat in settings.features:
                out_tmp_spline = self.spline[feat].score_samples(samples=processed_data_df[feat])
                out_tmp_kde = self.kde[feat].score_samples(processed_data_df[feat].values.reshape(-1, 1))
                out_spline += out_tmp_spline
                out_kde += out_tmp_kde
            assert not out_kde.any() <= 0
            assert not out_spline.any() <= 0
        else:
            raise Exception("Unsound IF/ELSE clause!")

        return {"df": processed_data_df, "res_main": out_spline, "res_sup": out_kde}

    def calculate_d(self, data_to_classify_csv, delta_baseline, truncate):
        """
        Calc delta as described in the paper
        :return:
        """
        self.logger.info('Kernel Density Classifier: \tCalculating delta with truncate ' + str(truncate))
        self.logger.info('Kernel Density Classifier: \tBaseline: ' + str(delta_baseline))

        # load data
        df_baseline = self.classify(delta_baseline)
        self.logger.info('Kernel Density Classifier: \tBaseline has size: ' + str(df_baseline.shape[0]))
        df_test = self.classify(data_to_classify_csv, truncate)
        intra_df_baseline = abs(df_baseline.max() - df_baseline.min())
        print("Intra df: " + str(intra_df_baseline))
        res = np.absolute(df_test - (df_baseline.min())) / intra_df_baseline
        return res.min(), res.max(), np.median(res)
