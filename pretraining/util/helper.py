import logging
import os
import os.path
import pandas as pd
import pickle
import re
import sys

import matplotlib.pyplot as plt

# define new logging levels, such as HYPER_PARAM_TUNING, which is between
# the DEBUG and INFO level.
HYPER_PARAM_TUNING = 25
logging.addLevelName(HYPER_PARAM_TUNING, "DEBUGV")


# overwrite debugv method of Logger
def debugv(self, message, *args, **kws):
    if self.isEnabledFor(HYPER_PARAM_TUNING):
        self._log(HYPER_PARAM_TUNING, message, args, **kws)


logging.Logger.debugv = debugv


# Logger factory
def create_logger(logger_level):
    """
    Returns a logger to be used by the main program
    :param logger_level     str or int, level of log
    """
    logger = logging.getLogger('L1')
    logger.setLevel(logger_level)
    ch = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(ch)

    return logger


def save_fig(figure, save_to_path, width=30, height=20, dpi=100):
    """
    Save a figure to a given path
    :param figure:              figure to save
    :param save_to_path:        path to save to
    :param width:               optional, image width
    :param height:              optional, image height
    :return:
    """
    figure.set_size_inches(width, height)
    # figure out ending of filename
    if save_to_path.endswith("png"):
        format = "png"
    elif save_to_path.endswith("eps"):
        format = "eps"
    else:
        raise ValueError("Format unspecified: " + save_to_path)
    if not os.path.exists(os.path.dirname(save_to_path)):
        os.makedirs(os.path.dirname(save_to_path))
    plt.savefig(save_to_path, dpi=dpi, format=format)


def purge(logger, dir, pattern):
    """
    Remove all files with pattern from directory
    :param dir:
    :param pattern:
    :return:
    """
    purge_count = 0
    for f in os.listdir(dir):
        if re.search(pattern, f) or f.endswith(pattern):
            if os.path.isfile(os.path.join(dir, f)):
                os.remove(os.path.join(dir, f))
                purge_count += 1
    logger.info("helper.py:\t\t\t\t\tRemoved {} files of type {}.".format(purge_count, pattern))


def _get_pickle(settings, path):
    """
    Read the pickle file and return its results
    :param settings:        BatchConfig Object, contains the current settings like data paths
    :param path:            str, path to folder containing pickle file
    :return:                content of pickle file
    """
    pickle_file_path = os.path.join(path, settings.nodes_dict_name)
    assert os.path.isfile(pickle_file_path), "Pickle Dictionary containing the nodes does not exist. " \
                                             "Did you run with feature_extraction flag to create it?" \
                                             "It is created during the feature extraction process." \
                                             "If you are unable to create it, ensure you run the simulation with AD" \
                                             "activated in the makefile.\n{}".format(pickle_file_path)
    with open(pickle_file_path, 'rb') as f:
        df_dictionary_keys = pickle.load(f)
    return df_dictionary_keys


def _find_best_model(number_neighbors, pretrain_folder):
    """
    Finds the best suited model for a given number of neighbors or raises an error if non is found.
    :param number_neighbors:            int, number of neighbors of the node
    :param pretrain_folder:             str, path to the folder containing the pretrained models
    :param train_data_path:             str, path to the current trainings data path
    :return:                            str, path to the best model
    """
    # Search for a model, a max range of 12 has been selected based on the size of the training networks
    for distance in range(0, 13):
        # Check for classifiers with less neighbors
        classifier_path = os.path.join(os.getcwd(), pretrain_folder,
                                       str(int(number_neighbors) - distance) + ".pkl")
        if os.path.isfile(classifier_path):
            # Early return in case we have a classifier
            return classifier_path

        # Check for classifiers with more neighbors
        classifier_path = os.path.join(os.getcwd(), pretrain_folder,
                                       str(int(number_neighbors) + distance) + ".pkl")
        if os.path.isfile(classifier_path):
            # Early return in case we have a classifier
            return classifier_path

    # If loop runs through, no model has been found, thus an error will be raised
    raise AssertionError('ERROR: No data found for {}. Last checked path was {}.'.format(pretrain_folder, classifier_path))


def get_positives(classification_results, threshold):
    """
    Calculates the percent of data points in classification_results, which are at most threshold.
    :param classification_results:          DataFrame, Contains the scores of a classifier
    :param threshold:                       float, Every data point with a score equal or lower to this value will be considered a positive
    :return:                                float, The fraction of positives to total number of data points
    """
    number_classifications = classification_results.shape[0]
    positives = classification_results[classification_results <= threshold]
    return float(positives.shape[0])/number_classifications


def load_threshold(pretrain_folder, threshold_file="threshold.txt"):
    """
    Loads the threshold stored in pretrain_folder/threshold_file.
    :param pretrain_folder:
    :return:
    """
    path = os.path.join(pretrain_folder, threshold_file)
    assert os.path.exists(path=path), "ERROR: {} does not exist. Please run run_calculate_threshold first " \
                                      "or check if path ad file are valid.".format(path)

    with open(path, 'r') as file:
        content = file.readline()

    threshold = float(content.split(': ')[1])
    return threshold


def _insert_without_ad(path):
    """
    Inserts 'without_AD' into a path
    :param path:            string, Path to a file
    :return:                string, Path with 'without_AD' inserted
    """
    directories = []

    # Find where to insert
    while True:
        path, directory = os.path.split(path)
        directories.append(directory)

        if directory == 'RPL-Attacks-Data' or directory is '':
            break

    # Reverse order so we can conrrectly build the path
    directories.reverse()

    # Build and return the modified path
    return os.path.join(path, directories[0], 'without_AD', *directories[1:])


def _save_data_as_table(path, data, in_percent=True):
    """
    Stores a data frame so it can easier be used as a table in a latex document.
    The data frame can only contain numerical values
    :param path:                string, Path were to store the table
    :param data:                DataFrame, Contains the data for the table
    :param in_percent:          boolean, True, if numerical values should be given in percent
    :return:                    None
    """
    # Define the symbols for separating columns and lines
    sep = '\t& '
    line_terminator = '\t\\\\ \\hline\n'

    # Start by getting the name of the index and the columns for the head line of the table
    table_string = data.index.name
    if table_string is None:
        table_string = ""

    for column in data.columns:
        table_string += sep + column
    table_string += line_terminator

    # Iterate over data frame and write the values in the string representing the table
    for index in data.index:
        table_string += index
        for column in data.columns:
            if in_percent:
                table_string += sep + "{:.4}\%".format(data.loc[index, column]*100)
            else:
                table_string += sep + "{}".format(data.loc[index, column])
        table_string += line_terminator

    # Store the table in the defined path
    with open(path, 'w') as file:
        file.write(table_string)


def collect_data(settings, condition_function, drop_first_n=0):
    """

    :param settings:
    :param condition_function:
    :param drop_first_n:
    :return:
    """
    # Dictionary for saving
    dic = {}

    # Collect data in a dict
    for data_set in settings.data_sets:
        settings.logger.log(level=40, msg="")
        # Iterate over all data sets, which do not contain the adversary but are not used for validation
        for train_data in set([s for s in settings.relative_test_data_paths if condition_function(s)]):
            settings.logger.log(level=40, msg="Using data from:\t{}".format(os.path.join(data_set, train_data)))
            current_path = os.path.join(data_set, train_data, "individual")
            for node in _get_pickle(settings, current_path):
                file = os.path.join(os.getcwd(), current_path, node + ".csv")
                dic = _append_file_to_dict(file=file, dic=dic, drop_first_n=drop_first_n,
                                           neighbors_feature=settings.neighbors_feature)

    return dic


def _append_file_to_dict(file, dic, drop_first_n, neighbors_feature):
    """
    Take a path to a file, read it and convert to DataFrame.
    Drop first n entries, then append it to the dict under the key "number_of_neighbors"
    :param file:                str, path to file
    :param dic:                 dict, dictionary for storing results, will be changed during processing
    :param drop_first_n:        int, specify how many entries should be dropped at the beginning of the data file
    :param neighbors_feature:   str, name of the feature containing the number of neighbors
    :return:                    {number neighbors: DataFrame}, the input dictionary, now containing the content of file
    """
    # Read file
    df = pd.read_csv(file, sep='\t')
    df = df.iloc[drop_first_n:]
    # df = _apply_percentile(df, .05, .95)

    # Group file by number of neighbors
    df_grouped = df.groupby(by=neighbors_feature)

    # Iterate over number of neighbors
    for num_neighbors in df_grouped.groups:
        # Add data points with num_neighbors neighbors to corresponding entry in dic
        statistics = df_grouped.get_group(num_neighbors).drop(neighbors_feature, axis='columns')
        if num_neighbors in dic:
            dic[int(num_neighbors)] = dic[int(num_neighbors)].append(statistics)
        else:
            dic[int(num_neighbors)] = statistics

    return dic
