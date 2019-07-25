import copy
import numpy as np
from io import BytesIO
from subprocess import check_output

import os
import pandas as pd

from pretraining.processing.log_parser import process_log_file, _process_fucntion_time_content, extract_edge_weights
from pretraining.util.helper import _insert_without_ad, _get_pickle, _save_data_as_table, _find_best_model, \
    load_threshold, get_positives, _append_file_to_dict


def get_function_time_statistics(settings):
    """

    :param settings:
    :return:
    """
    path_to_data_set = settings.data_sets[0]
    data_frame_list = []
    for relative_path in settings.relative_test_data_paths:
        path = os.path.join(path_to_data_set, relative_path, 'serial.log')
        data_frame = process_log_file(path_to_log_file=path, separator_tuple=('[Q]',),
                                      feature_processing_function=_process_fucntion_time_content)['[Q]']
        data_frame_list.append(data_frame)

    complete_frame = pd.concat(data_frame_list, ignore_index=True)
    mean = complete_frame.mean(axis=0)
    max = complete_frame.max(axis=0)
    min = complete_frame.min(axis=0)
    statistics = pd.concat([mean, max, min], axis=1)
    statistics.columns = ['mean', 'max', 'min']

    save_path = os.path.join(settings.global_pretrain_folder, 'function_time_statistics.csv')

    statistics.to_csv(save_path)


def get_size_statistics(settings):
    """
    Generates statistics for the size of contiki with AD and without.
    :param settings:
    :return:
    """

    without_ad_data_sets = [_insert_without_ad(data_set) for data_set in settings.data_sets]
    with_ad_data_sets = settings.data_sets

    path_without = os.path.join(without_ad_data_sets[0], "control-set-clean/without-malicious/motes/sensor.z1")
    path_with = os.path.join(with_ad_data_sets[0], "control-set-clean/without-malicious/motes/sensor.z1")

    output_without = check_output(['size', path_without])
    output_with = check_output(['size', path_with])

    df_without = pd.read_csv(BytesIO(output_without), sep='\t')
    df_with = pd.read_csv(BytesIO(output_with), sep='\t')

    df_combined = df_without.append(df_with, ignore_index=True)
    df_combined.columns = map(lambda x: x.strip(), df_combined.columns)

    df_combined.index = ["without", "with"]
    df_combined.index.name = 'AD'

    df_combined = df_combined.drop(labels=["hex", "filename"], axis=1)

    save_path = os.path.join(settings.global_pretrain_folder, "size.table")

    _save_data_as_table(path=save_path, data=df_combined[["text", "data", "bss", "dec"]], in_percent=False)


def _evaluate_positives(settings, data_dict):
    """

    :param settings:
    :return:
    """
    results = None

    settings = copy.copy(settings)

    settings.run_generate_data_model_default()

    for model in settings.model_list:
        # Create a model for each number of neighbors
        for num_neighbors in data_dict.keys():

            # Concat all dataframes for a given number of neighbors into a single one
            data_frame = data_dict[num_neighbors]#pd.concat(dic[num_neighbors], axis=0)
            data_frame = data_frame.reset_index(drop=True)

            # Train a model on the complete data (this takes a lot of time)
            # Find best classifer for each number of neighbors
            classifier_path = _find_best_model(number_neighbors=num_neighbors,
                                               pretrain_folder=settings.global_pretrain_folder)
            # Load the best classifier
            model.load_pretrained_model(classifier_path=classifier_path)

            if results is None:
                results = model.classify_on_data(df=data_frame, settings=settings)["res_main"]
            else:
                results = np.concatenate((results, model.classify_on_data(df=data_frame, settings=settings)["res_main"]), axis=0)

    threshold = load_threshold(settings.global_pretrain_folder)
    settings.set_threshold(threshold)

    return get_positives(classification_results=results, threshold=threshold)


def _get_generic_positves(settings, data_set_limiter):
    """

    :param settings:
    :param data_set_limiter:
    :return:
    """
    drop_first_n = settings.drop_first_n_for_positives
    # Dictionary for saving
    data_dict = {}

    # Collect data of nodes, who can detect the anomaly in a dict
    for data_set in settings.data_sets:
        settings.logger.log(level=40, msg="")
        # Iterate over all data sets, which do not contain the adversary but are not used for validation
        for train_data in set([s for s in settings.relative_test_data_paths if "with-" in s and data_set_limiter in s]):
            settings.logger.log(level=40, msg="Using data from:\t{}".format(os.path.join(data_set, train_data)))
            neighborhood = _get_mal_neighborhood(path=os.path.join(data_set, train_data), logger=settings.logger)
            current_path = os.path.join(data_set, train_data, "individual")
            for node in _get_pickle(settings, current_path):
                if neighborhood.isin([float(node)]).any():
                    file = os.path.join(os.getcwd(), current_path, node + ".csv")
                    data_dict = _append_file_to_dict(file=file, dic=data_dict, drop_first_n=drop_first_n,
                                                     neighbors_feature=settings.neighbors_feature)

    true_positives = _evaluate_positives(settings=settings, data_dict=data_dict)
    return true_positives


def _get_mal_neighborhood(path, logger):
    """

    :param path:
    :param logger:
    :return:
    """
    package_count = extract_edge_weights(path_to_log_file_folder=path, logger=logger)
    packets_to_mal = package_count[package_count["To"]==np.max(package_count["To"])]
    sender_to_mal = packets_to_mal["From"]

    return sender_to_mal


def calc_statistics():
    """

    :return:
    """
    # Load data frame
    load_path = os.path.join(os.getcwd(), 'global_train', 'function_time_statistics.csv')

    df = pd.read_csv(load_path, index_col=0)

    # Variables for average features
    poll_all_features_avg = df.loc[' poll_all_features', 'mean']
    update_neighbor_counts_avg = df.loc[' update_neighbor_counts', 'mean']
    sliding_window_forward_avg = df.loc[' sliding_window_forward', 'mean']
    reset_dodag_version_bitarray_avg = df.loc[' reset_dodag_version_bitarray', 'mean']
    increment_cc_array_position_avg = df.loc[' increment_cc_array_position', 'mean']


    # Variables for max features
    poll_all_features_max = df.loc[' poll_all_features', 'max']
    update_neighbor_counts_max = df.loc[' update_neighbor_counts', 'max']
    sliding_window_forward_max = df.loc[' sliding_window_forward', 'max']
    reset_dodag_version_bitarray_max = df.loc[' reset_dodag_version_bitarray', 'max']
    increment_cc_array_position_max = df.loc[' increment_cc_array_position', 'max']

    # Aligning in a list
    averages = [poll_all_features_avg, update_neighbor_counts_avg,
                sliding_window_forward_avg, sliding_window_forward_avg, sliding_window_forward_avg,
                reset_dodag_version_bitarray_avg, increment_cc_array_position_avg]
    max_vals = [poll_all_features_max, update_neighbor_counts_max,
                sliding_window_forward_max, sliding_window_forward_max, sliding_window_forward_max,
                reset_dodag_version_bitarray_max, increment_cc_array_position_max]

    # Weights are the frequency of the calls
    weights = [0.2, 1/15,
               1,1,1,
               1/500, 1/500]

    mean = np.sum(np.array(averages)*np.array(weights))
    max = np.sum(np.array(max_vals)*np.array(weights))

    print('Ticks per second:\n')
    print('mean: {}, max: {}\n\n'.format(mean, max))

    anomaly_detection_main_loop_avg = df.loc[' anomaly_detection_main_loop', 'mean']
    anomaly_detection_main_loop_max = df.loc[' anomaly_detection_main_loop', 'max']

    print('Initialization cost:\n')
    print('mean: {}, max: {}\n\n'.format(anomaly_detection_main_loop_avg, anomaly_detection_main_loop_max))

    network_stack_part_avg = df.loc[' network_stack_part', 'mean']

    network_stack_total_avg = df.loc[' network_stack_total', 'mean']

    network_stack_orig_avg = network_stack_total_avg - network_stack_part_avg

    print('Average time per packet:\n')
    print('total: {}, part: {}: orig:{}, increase: {}'.format(network_stack_total_avg, network_stack_part_avg,
                                                              network_stack_orig_avg, network_stack_total_avg/network_stack_orig_avg))