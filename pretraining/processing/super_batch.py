import matplotlib

matplotlib.use('Agg')

import glob
import random
import shutil
from time import gmtime, strftime
import multiprocessing
import os
import copy
import pandas as pd
import warnings

print("\nDANGER MODE: IGNORING ALL WARNINGS!\n")
warnings.filterwarnings('ignore')


def run_super_batch(settings, func, multiprocess_features, multiprocess_datasets):
    """
    Runs a single task multiple times. The data sets, classifiers, features and attacks have to be specified in a
    settings object. The function determines which task should be run multiple times.
    This allows to run a whole batch of related computations in one fell swoop.
    :param multiprocess_datasets:   True if divide datasets on available cores
    :param multiprocess_features:   True if divide features on available cores
    :param settings:                batch_config object, An object containing the global settings
    :param func:                    function(batch_config), A function taking only a batch_config object as parameter and
                                    returning a String
    :return:                        None
    """
    # XOR
    assert not (multiprocess_features and multiprocess_datasets), "Choose either feat- or dataset multiprocessing!"

    # Getting the number of cpus
    num_cores = multiprocessing.cpu_count() if (multiprocess_features or multiprocess_datasets) else 1

    # check length of inputs
    if len(settings.feature_combinations) < 1:
        raise ValueError("Empty feature set!")
    if len(settings.data_sets) < 1:
        raise ValueError("Empty Test Data set!")

    # create interm output dir
    if os.path.exists(settings.folder_for_parial_interm_results):
        settings.logger.log(level=20, msg="SuperBatch.py:\t\tRemoving old intermediate directory {}".format(
            settings.folder_for_parial_interm_results))
        shutil.rmtree(settings.folder_for_parial_interm_results)
    if not os.path.exists(settings.folder_for_parial_interm_results):
        settings.logger.log(level=20, msg="SuperBatch.py:\t\tRecreate intermediate directory {}".format(
                                settings.folder_for_parial_interm_results))
        os.makedirs(settings.folder_for_parial_interm_results)

    # ==================== MULTIPROCESSING ====================
    settings.logger.log(level=20, msg="SuperBatch.py:\t\t{} cores available!".format(num_cores))

    # Create dict used for saving the processes
    process_dict = {}

    # Start the processes
    if multiprocess_features or num_cores == 1:
        # SPLIT ON FEATURES
        for feature_set, process_id in list_slice_generator(list=settings.feature_combinations, number_of_slices=num_cores):
            settings_copy = copy.copy(settings)
            settings_copy.feature_combinations = feature_set
            p = multiprocessing.Process(target=_run_model_path_loops, args=(func, settings_copy, process_id))
            process_dict[process_id] = p
            p.start()
    elif multiprocess_datasets:
        # SPLIT ON DATA

        # create a list of data sets.
        datasets_full = []
        for path in settings.data_sets:
            for subpath in settings.relative_test_data_paths:
                datasets_full.append(os.path.join(os.getcwd(), path, subpath, settings.test_file_name))
        # start processes
        for data_set, process_id in list_slice_generator(list=datasets_full, number_of_slices=num_cores):
            settings_copy = copy.copy(settings)
            p = multiprocessing.Process(target=_run_model_path_loops,
                                        args=(func, settings_copy, process_id, True, data_set))
            process_dict[process_id] = p
            p.start()
    else:
        raise ValueError("Code should never reach this!")

    # Wait until all processes have finished
    for process_id in process_dict:
        process_dict[process_id].join()

    # aggregate partial files into one big file:
    read_files = glob.glob(os.path.join(settings.folder_for_parial_interm_results, "*"))
    with open(settings.save_interm_results, "a+") as outfile:
        for f in read_files:
            with open(f, "rb") as infile:
                outfile.write(infile.read())

    settings.logger.log(level=20, msg="Wrote intermediate results file to {}".format(
        os.path.join(os.getcwd(), settings.save_interm_results)))


def _run_model_path_loops(func, settings_copy, process_id, split_on_dataset=0, data_set=None):
    """
    Function to be run by a single core
    :param func:
    :param settings_copy:
    :return:
    """
    assert (data_set is None) == (1 != split_on_dataset), "Must not enter a data set if split_on_data==0"

    progress_count = 0
    for features in settings_copy.feature_combinations:
        progress_count += 1
        if progress_count % 10 == 0:
            settings_copy.logger.log(level=50, msg="Process {}: {}%".format(
                process_id, 100. * progress_count / len(settings_copy.feature_combinations)))

        # Set the features. Currently the features are still read from the global settings.
        settings_copy.features = features

        # Make only one feature string, which is used always
        feature_string = create_feature_number_str(features, settings_copy.features_to_number_dict)

        # We have to reset the model generator so we can have our models in the next run
        settings_copy.model_list = settings_copy.model_generator()

        # Create DF to store results from individual iterations if they return anything
        r_batch = pd.DataFrame()

        # Iterate over classifiers
        for classifier in settings_copy.model_list:
            # Set the current model for training
            settings_copy.model = classifier
            # Iterate over all data sets
            if not split_on_dataset:
                for data_set_path in settings_copy.data_sets:
                    settings_copy.logger.log(level=40,
                                             msg=strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " \tCore {}\t{}\t{}\t{}."
                                             .format(process_id, data_set_path, classifier, features))
                    # Create the absolute training data path
                    settings_copy.training_data_path = os.path.join(data_set_path,
                                                                    settings_copy.relative_training_data_path)
                    settings_copy.current_data_set_path = data_set_path

                    assert settings_copy.training_data_path is not None

                    # iterate over arguments in list
                    for rel_test_data_path in settings_copy.relative_test_data_paths:
                        settings_copy.test_data_path = os.path.join(settings_copy.current_data_set_path,
                                                                    rel_test_data_path,
                                                                    settings_copy.test_file_name)

                        # Call func and append output if exists
                        r_batch_tmp = func(settings=settings_copy)
                        if isinstance(r_batch_tmp, pd.DataFrame) and not r_batch_tmp.empty:
                            r_batch.append(r_batch_tmp)

                    settings_copy.logger.log(level=40, msg=strftime("%Y-%m-%d %H:%M:%S", gmtime()) +
                                             " \t\tCore {} done on current job!".format(process_id))

            # if not split_on_dataset, e.g. if we do not parallelize computation on data sets
            else:
                # Run a batch. This calculates the function on every test set.
                for x in data_set:
                    settings_copy.test_data_path = x
                    settings_copy.logger.log(level=40,
                                             msg=strftime("%Y-%m-%d %H:%M:%S", gmtime()) + "\t\tp{}\t\t{}\t\t{}\t\t{}."
                                             .format(process_id, settings_copy.test_data_path, classifier, features))

                    # Call func and append output if exists
                    r_batch_tmp = func(settings=settings_copy)
                    if isinstance(r_batch_tmp, pd.DataFrame) and not r_batch_tmp.empty:
                        r_batch.append(r_batch_tmp)

                settings_copy.logger.log(level=40,
                                         msg=strftime("%Y-%m-%d %H:%M:%S", gmtime()) + " \t\tCPU {} done".format(process_id))

        # Write to file after processing a data set
        if not r_batch.empty:
            r_batch.to_csv(os.path.join(settings_copy.folder_for_parial_interm_results, feature_string), sep='\t')
            settings_copy.logger.log(level=20, msg="Wrote intermediate results to {}".format(
                os.path.join(os.getcwd(), settings_copy.folder_for_parial_interm_results, feature_string)))


def list_slice_generator(list, number_of_slices):
    """
    A generator which returns the slices of a list. This allows to split the list in more even parts.
    :param list:                list, The list to split
    :param number_of_slices:    int, The number of slices
    :return:                    (list, int), a list slice and the number of the slice in the list
    """
    # Some sanity checks
    assert number_of_slices > 0 and number_of_slices == int(number_of_slices), \
        "The number of slices has to be a positive integer and not {}".format(number_of_slices)
    assert list is not None and len(list) > 0, "The list should not be None or empty"

    # shuffle list
    random.shuffle(list)

    length = len(list)
    # Setting the effective number of slices
    if number_of_slices > length:
        effective_number_of_slices = length
    else:
        effective_number_of_slices = number_of_slices
    # Values to keep track of progress
    step_size = float(length) / effective_number_of_slices
    current_value = 0.
    chunk_number = 0

    while current_value < length:
        # Rounding and floats are used for a more even split
        yield list[int(round(current_value)):int(round(current_value + step_size))], chunk_number
        current_value += step_size
        chunk_number += 1


def create_feature_number_str(feature_list, feature_to_number_dict):
    """

    :param feature_list:
    :param feature_to_number_dict:
    :return:
    """
    feature_str = ''
    for feature in feature_list:
        feature_str += feature_to_number_dict[feature] + ', '

    return feature_str
