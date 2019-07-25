import copy
import os
import numpy as np
import pandas as pd
from scipy.stats import binom

from pretraining.util.helper import load_threshold, _get_pickle, _find_best_model, get_positives, _save_data_as_table, \
    _append_file_to_dict
from pretraining.processing.log_parser import extract_edge_weights


def calculate_positive_probability(settings):
    """
    Calculates the number of positives for each test set using test theory.
    :param settings:                batch_config object, Settings of the module
    :return:                        None
    """
    drop_first_n = settings.drop_first_n

    # Generate a copy, so we are less likely to unintentionally mutate something
    settings = copy.copy(settings)

    # Ensure the right settings
    settings.run_generate_data_model_default()

    # Load and set the current threshold
    threshold = load_threshold(settings.global_pretrain_folder)
    settings.set_threshold(threshold)

    # Define the significance
    significance = 0.01

    # A data frame for storing the results
    result_frame = pd.DataFrame()
    count_frame = pd.DataFrame()
    percent_frame = pd.DataFrame()
    times_average_frame = pd.DataFrame()

    # Iterate over all the models
    for model in settings.model_list:

        # Iterate over all data sets
        for data_set in settings.data_sets:

            # Set logging level
            settings.logger.log(level=40, msg="")

            # Iterate over all sub data sets, except the training data
            for train_data in [path for path in settings.relative_test_data_paths
                               if "control-set-clean" in path or "with-malicious" in path]:
                # Logging to keep track, where we are
                settings.logger.log(level=40, msg="Using data from:\t{}".format(os.path.join(data_set, train_data)))
                current_path = os.path.join(data_set, train_data)

                # Store the positives in the result data frame
                column_name_dict = {"control-set-clean/without-malicious/data": "\\thead{FP}",
                                    "blackhole-attack/with-malicious/data": "\\thead{TP\\\\Blackhole}",
                                    "hello-flood-attack/with-malicious/data": "\\thead{TP\\\\H. Flood}",
                                    "version-number-attack/with-malicious/data": "\\thead{TP\\\\Vers. Num.}"}

                positives = test_dataset(os.path.join(current_path, "individual"), drop_first_n, model, settings,
                                         significance, threshold)

                if train_data in column_name_dict:
                    result_frame.loc[os.path.split(data_set)[1][:-1], column_name_dict[train_data]] = positives

                from_to_count_frame = extract_edge_weights(path_to_log_file_folder=current_path, logger=settings.logger)
                to_count_frame = from_to_count_frame.drop("From", axis=1)
                to_sum_frame = to_count_frame.groupby("To").sum()

                if "blackhole" in train_data:
                    malicious_node = to_count_frame["To"].max()

                    sum_to_malicious = to_sum_frame.loc[malicious_node][0]
                    sum_overall = to_sum_frame.sum()[0]

                    percent_of_all = float(sum_to_malicious)/sum_overall

                    mean_to = to_sum_frame.mean()[0]
                    times_average = float(sum_to_malicious)/mean_to - 1

                    if train_data in column_name_dict:
                        count_frame.loc[os.path.split(data_set)[1][:-1], column_name_dict[train_data]] = sum_to_malicious
                        percent_frame.loc[os.path.split(data_set)[1][:-1], column_name_dict[train_data]] = percent_of_all
                        times_average_frame.loc[os.path.split(data_set)[1][:-1], column_name_dict[train_data]] = times_average
                        result_frame.loc[os.path.split(data_set)[1][:-1], "\\thead{Times\\\\Average}"] = times_average

        # Add average row and column
        result_frame.loc["All", :] = result_frame.mean(axis=0)

        # Store results
        table_store_path = os.path.join(settings.global_pretrain_folder, "TP_FP_results.table")
        _save_data_as_table(path=table_store_path, data=result_frame, in_percent=True)
        print(result_frame)
        print(count_frame)
        print(percent_frame)
        print(times_average_frame)


def test_dataset(dataset_path, drop_first_n, model, settings, significance, threshold):
    """
    Test a single dataset and return the percentage of positives.
    :param dataset_path:                string, Path to the data set
    :param drop_first_n:                int, Number of how many data points at the start of a data set should be dropped
    :param model:                       base_classifier, The model used for predictions
    :param settings:                    batch_config object, Object for storing settings
    :param significance:                float, Significance level of the hypothesis test
    :param threshold:                   float, Value at which a single node considers an event an anomaly
    :return:                            float, Percentage of positives in the data set
    """
    # Dictionary for storing the results of the individual nodes
    data_set_result_dict = {}
    # Dictionary containing the nodes in the network except from the root and the adversary
    node_dictionary = _get_pickle(settings, dataset_path)
    # Increment the number of nodes, since the root is not contained in the dictionary
    number_nodes_in_dataset = len(node_dictionary) + 1
    # Calculate at which number of alarms it is considered an anomaly
    decision_boundary = get_decision_boundary(number_nodes=number_nodes_in_dataset,
                                              false_positive_probability=0.01, significance=significance)
    # Extract data for each node individually
    for node in node_dictionary:
        # Dictionary for storing data of the node based on the number of neighbors the node has at his point in time
        dic = {}

        # Load the data from the file and store it in the dictionary
        file = os.path.join(os.getcwd(), dataset_path, node + ".csv")
        dic = _append_file_to_dict(file=file, dic=dic, drop_first_n=drop_first_n,
                                   neighbors_feature=settings.neighbors_feature)

        # Initialize variable for storing the entire data for the node
        single_dataset_result_frame = None

        # Load a model for each number of neighbors
        for num_neighbors in dic.keys():

            # Concat all dataframes for a given number of neighbors into a single one
            data_frame = dic[num_neighbors]  # pd.concat(dic[num_neighbors], axis=0)

            # Store the index so we can restore it later
            idx = data_frame.index
            data_frame = data_frame.reset_index(drop=True)

            # Find best classifier for each number of neighbors
            classifier_path = _find_best_model(number_neighbors=num_neighbors,
                                               pretrain_folder=settings.global_pretrain_folder)
            # Load the best classifier
            model.load_pretrained_model(classifier_path=classifier_path)

            if num_neighbors != 0:
                classification_result = pd.DataFrame(model.classify_on_data(
                        df=data_frame, settings=settings)["res_main"], index=idx)
            else:
                classification_result = pd.DataFrame(np.zeros(data_frame.shape), index=idx)

            # Store the classifier score and restore the index
            if single_dataset_result_frame is None:
                single_dataset_result_frame = classification_result
            else:
                single_dataset_result_frame = pd.concat((single_dataset_result_frame, classification_result), axis=0)

        # Determine the classification based on the score and the threshold, store the classification results sorted by index
        classification_result_dataframe = single_dataset_result_frame <= threshold
        data_set_result_dict[node] = classification_result_dataframe.astype(int).sort_index()
    # Combine the DataFrames of all individual motes and count the number of alarms at a given time
    all_nodes_in_dataset_results = pd.concat(data_set_result_dict.values(), axis=1)
    number_alarms_dataframe = all_nodes_in_dataset_results.sum(axis=1)
    # Calculate positives using a threshold, here get positives actually gives the negatives
    positives = 1 - get_positives(classification_results=number_alarms_dataframe, threshold=decision_boundary)
    return positives


def binomial_test_probability(false_positive_probability, number_nodes, dataframe):
    """
    Given a data frame containing the number of occurrences calculate P(X>=k) for each row assuming
    X ~ Bin(number_nodes, false_positive_probability)
    :param false_positive_probability:          float, The probability of false positives occurring in the network
    :param number_nodes:                        int, The number of nodes deployed in the network
    :param dataframe:                           DataFrame, Contains the number of nodes which conclude that an anomaly has occurred
    :return:                                    DataFrame, Contains for each row the probability P(X>=k) under the binomial model
    """
    probability_data_frame = dataframe.map(lambda x: 1 - binom.cdf(k=x, n=number_nodes, p=false_positive_probability) +
                                                     binom.pmf(k=x, n=number_nodes, p=false_positive_probability))
    return probability_data_frame


def get_decision_boundary(number_nodes, false_positive_probability, significance):
    """
    Calculates min(c in Natural numbers) where P(X>=c)<=significance with
    X ~ Binomial(number_nodes, false_positive_probability). c is at least 2 since it is otherwise impossible to distinguish
    some anomalies from false positives.
    We subtract one, since we want to return the highest value at which it still isn't an anomaly.
    :param number_nodes:                        int, The number of nodes in the network
    :param false_positive_probability:          float, probability of a data point being a false positive
    :param significance:                        float, Significance for the binomial test
    :return:                                    int, max(1, c-1)
    """
    # Initialize the probability of P(X>=k) for k=0
    p_x_geq_k = 1
    k = 0

    # Search for the minimum c with P(X>=c)<=significance
    while p_x_geq_k > significance:
        p_x_geq_k -= binom.pmf(k=k, n=number_nodes, p=false_positive_probability)
        k += 1

    # Return max(1, c-1) to reduce the number of false positives in case c = 1,
    return max(1, k-1)
