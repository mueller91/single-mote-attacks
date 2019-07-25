import pandas
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, MinMaxScaler


class BaseClassifier:
    """Abstract base class for all classifiers, such that they
    share a common interface"""

    def __init__(self, logger, n_dim_reduction, features_to_select=None):
        self.normalizer = Normalizer(norm="l2")
        self.minmax_scaler = MinMaxScaler()
        self.n_dim_reduction = n_dim_reduction
        self.dim_reduction_pca = PCA(n_components=n_dim_reduction) if n_dim_reduction else None
        self.features_to_select = features_to_select
        self.logger = logger

    def train_on_csv(self, training_data_file_csv, settings):
        """
        Training method, to be overwritten by subclass.
        :param training_data_file_csv:          String, path to training data
        :param settings:                        batch config object, Configuration object
        :return:                                None
        """
        raise NotImplementedError()

    def train_on_data(self, df, settings):
        """
        Method for training on data. Has to be implemented in sub class.
        :param settings:        batch config object, Configuration object
        :param df:              DataFrame, Contains data
        :return:                None
        """
        raise NotImplementedError

    def classify(self, data_to_classify_csv, settings):
        """
        Classify method, to be overwritten by subclass
        :param settings:
        :param data_to_classify_csv:    String, containing path to input file
        :return:                        Array A, where A[i] == 1 iff log batch i was classified as normal
        """
        raise NotImplementedError()

    def classify_on_data(self, df, settings):
        """

        :param settings:
        :param df:
        :return:
        """
        return NotImplementedError()

    def sanetize_select(self, df, settings):
        """
        Do basic sanitizing: Remove legacy columns, check if df != null
        :param trained_on:
        :param settings
        :param df:
        :return:
        """
        assert not df.empty, "Dataframe empty:\n{}".format(df)

        # Fix mismatch trained/training
        trained_on_but_not_in_df = set(settings.features).difference(df.columns)
        for x in trained_on_but_not_in_df:
            self.logger.log(level=50,
                            msg="Base Class:\t\t\t\t\tSetting feature {} to 0 because it's not in the training set!".format(
                                x))
            df[x] = 0
        df = df[settings.features].fillna(0)

        # select features
        if self.features_to_select:
            try:
                self.logger.info("Base Class:\t\t\t\t\tSelecting features {}".format(self.features_to_select))
                return df[self.features_to_select].astype(float)
            except KeyError:
                msg = "The following features are selected, but not present in the DF columns:\n" + \
                      "{}".format([x for x in self.features_to_select if x not in df.columns])
                raise KeyError(msg)

        else:
            # warn if self.features_to_select is off
            self.logger.warn(
                "Base Class:\t\t\t\t\tWarning:"
                "Not selecting any features - classification will run on complete feature map!")
            self.logger.info("BaseClass:\t\t\t\t\tNot selecting subset of features: Returning the whole features set.")
            return df.astype(float)

    def transform_data(self, df, settings):
        """
        Apply necessary transformations, based on boolean argument flags
        :param df:
        :param settings:
        :return:
        """

        # Log settings
        processed_data = df.values

        if settings.normalize:
            self.logger.info('Base Class: \t\t\t\tTransform normalisation!')
            processed_data = self.normalizer.transform(df.values)

        if settings.minmax_scaling:
            processed_data = self.minmax_scaler.transform(processed_data)

        if settings.dim_reduction and self.n_dim_reduction:
            self.logger.info('Base Class: \t\t\t\tTransform dim Reduction to n={}'.format(settings.dim_reduction))
            processed_data = self.dim_reduction_pca.transform(processed_data)

        return pandas.DataFrame(processed_data, columns=df.columns, index=df.index)

    def fit_transform_data(self, df, settings):
        """
        Apply necessary transformations, based on boolean argument flags
        :param settings:
        :param df:
        :return:
        """

        # Log settings
        if settings.normalize:
            self.logger.info('Base Class: \t\t\t\tFit_transform normalisation!')
        if settings.minmax_scaling:
            self.logger.info('Base Class: \t\t\t\tFit_transform MinMaxScaling!')
        if settings.dim_reduction:
            self.logger.info('Base Class: \t\t\t\tFit_transform dim Reduction to n={}'.format(settings.dim_reduction))

        # normalize fit / transform
        if settings.normalize:
            processed_data = self.normalizer.fit_transform(df.values)
        else:
            processed_data = df.values

        # minmax fit / transform
        if settings.minmax_scaling:
            processed_data = self.minmax_scaler.fit_transform(processed_data)
        else:
            pass

        # dimensionality reduction
        if settings.dim_reduction:
            processed_data = self.dim_reduction_pca.fit_transform(processed_data)
        else:
            pass

        return pandas.DataFrame(processed_data, columns=df.columns, index=df.index)
