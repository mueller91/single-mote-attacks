import os
import pickle

import pandas as pd

# Create variables for easy renaming in case features get renamed
anomaly_score = "ANOMALY SCORE"
id_name = "ID"
normalized_score = "Normalized Score"
feature_tag = "[CT]"
score_tag = "[SC]"
power_tag = " P "
flow_tag = "[UF] "
# The number of ticks per second depends on the hardware. For MSP430X it is 32768.
ticks_per_second = 32768


def _default_log_content_processing(msg_str, feature_dict):
    """
    The default processing mechanism of the content part of log files.
    It assumes it has the following structure:
    (feature_name: feature_value, )*feature_name: feature_value

    Note: feature_dict will be changed during processing
    :param msg_str:                 str, a string representing the features
    :param feature_dict:            dict, a dictionary to save results into
    :return:                        None
    """
    # Split content part in actual features
    msg = msg_str.split(',')

    # iterate over list of features
    for feature in msg:
        assert len(feature.split(': ')) == 2, "ERROR: should have to parts: {}".format(feature)
        feature_name, value = feature.split(': ')
        try:
            feature_dict[feature_name.strip()] = (float(value.strip()))
        except ValueError as value_error:
            print(value_error)


def _process_flow_content(msg_str, feature_dict):
    """

    :param msg_str:
    :param feature_dict:
    :return:
    """
    # Split content part in actual features
    msg = msg_str.split(';')[0].split(', ')

    # Iterate over list of features
    for feature in msg:
        assert len(feature.split('=')) == 2, "ERROR: should have to parts: {}".format(feature)
        feature_name, value = feature.split('=')
        feature_dict[feature_name.strip()] = (int(value.strip()))


def _process_fucntion_time_content(msg_str, feature_dict):
    """

    :param msg_str:
    :param feature_dict:
    :return:
    """
    func, time = msg_str.split(': ')
    feature_dict[func] = int(time)


def _process_powertracking_content(msg_str, feature_dict):
    """
    Processes the content part of lines containing powertrace information.
    All non-percentage values are given in ticks, which depends on hardware.
    In our case it is always 32768 ticks per second.
    The features are extracted and transformed into seconds, if given as ticks.
    0:  linkaddr_node_addr.u8[0], linkaddr_node_addr.u8[1],
    1:  seqno,
    2:  all_cpu,
    3:  all_lpm,
    4:  all_transmit,
    5:  all_listen,
    6:  all_idle_transmit,
    7:  all_idle_listen,
    8:  cpu,
    9:  lpm,
    10: transmit,
    11: listen,
    12: idle_transmit,
    13: idle_listen,
    15: (int)((100L * (all_transmit + all_listen)) / all_time),
        (int)((10000L * (all_transmit + all_listen) / all_time) - (100L * (all_transmit + all_listen) / all_time) * 100),
    17: (int)((100L * (transmit + listen)) / time),
        (int)((10000L * (transmit + listen) / time) - (100L * (transmit + listen) / time) * 100),
    19: (int)((100L * all_transmit) / all_time),
        (int)((10000L * all_transmit) / all_time - (100L * all_transmit / all_time) * 100),
    21: (int)((100L * transmit) / time),
        (int)((10000L * transmit) / time - (100L * transmit / time) * 100),
    23: (int)((100L * all_listen) / all_time),
        (int)((10000L * all_listen) / all_time - (100L * all_listen / all_time) * 100),
    25: (int)((100L * listen) / time),
        (int)((10000L * listen) / time - (100L * listen / time) * 100));

    Note: feature_dict will be altered during calculation and used for storing results.
    :param msg_str:                 str, produced by powertrace containing power information
    :param feature_dict:            dict, a dictionary for storing information, will be altered for calculation
    :return:                        None
    """
    # Split message
    msg = msg_str.split(' ')

    # Features with 'All' contain information of entire run time, other only about time since last print
    # All times are given in Seconds
    # Note: powertrace only provides the time in ticks, thus we need to calculate back to seconds.
    #       Since we are only using the MSP430X with a 32768 Hz quartz clock, we can use this constant for calculation.
    # feature_dict["Node Addresse"] = msg[0] # Not needed
    feature_dict["All CPU Time"] = float(msg[2])/ticks_per_second
    feature_dict["All LPM Time"] = float(msg[3])/ticks_per_second
    feature_dict["All Transmit Time"] = float(msg[4])/ticks_per_second
    feature_dict["All Listen Time"] = float(msg[5])/ticks_per_second
    feature_dict["All Idle Transmit Time"] = float(msg[6])/ticks_per_second
    feature_dict["All Idle Listen Time"] = float(msg[7])/ticks_per_second
    feature_dict["CPU Time"] = float(msg[8])/ticks_per_second
    feature_dict["LPM Time"] = float(msg[9])/ticks_per_second
    feature_dict["Transmit Time"] = float(msg[10])/ticks_per_second
    feature_dict["Listen Time"] = float(msg[11])/ticks_per_second
    feature_dict["Idle Transmit Time"] = float(msg[12])/ticks_per_second
    feature_dict["Idle Listen Time"] = float(msg[13])/ticks_per_second
    # These features contain an estimate of error after the '.', not needed here and thus dropped.
    feature_dict["All Radio Time in Percent"] = msg[15].split('.')[0]
    feature_dict["Radio Time in Percent"] = msg[17].split('.')[0]
    feature_dict["All Transmission Time in Percent"] = msg[19].split('.')[0]
    feature_dict["Transmission Time in Percent"] = msg[21].split('.')[0]
    feature_dict["All Listen Time in Percent"] = msg[23].split('.')[0]
    feature_dict["Listen Time in Percent"] = msg[25].split('.')[0]


def parse_to_frame(content, sep, feature_processing_function=_default_log_content_processing):
    """
    Parses a list of strings and returns the data as a pandas data frame
    :param content:                         [str], The list of strings, which should be parsed.
    :param sep:                             str, A separator used to identify lines to process
    :param feature_processing_function:     function, takes a msg_str and a dictionary, processes msg_str, stores results in dictionary
    :return:                                data frame, A pandas data frame containing all the feature information in content.
    """
    # Data frame  for storing intermediate data
    res_df = pd.DataFrame()

    # Iterate over all lines in content
    for line in content:
        # Check if sep is in line and thus, line should be parsed
        if sep in line:
            # Split line in id and content part
            node_id_str, msg_str = line.split(sep)

            # Create dictionary for storing feature information
            feature_dict = {}

            # Extract id from id part and store it in feature_dict
            node_id_fields = node_id_str.split(":")
            node_id = int(node_id_fields[1].strip().split('\t')[0])
            feature_dict[id_name] = node_id

            # Process features from content part
            feature_processing_function(msg_str, feature_dict)

            # Append features of line to data frame
            res_df = res_df.append(feature_dict, ignore_index=True)
        
    return res_df


def process_log_file(path_to_log_file, separator_tuple=(feature_tag, score_tag),
                     feature_processing_function=_default_log_content_processing):
    """
    Processes an entire log file and returns the data frames corresponding to it.
    :param path_to_log_file:            str, path to the log file
    :param separator_tuple:             tuple of strings, content will be used as separator/line tags for extracting data from the log file
    :param feature_processing_function: function, processes the part containing the features
    :return:                            {str: data frame}, key: str, separator,  value: data frame, containing data of log file
    """
    # Check if path exists
    assert os.path.exists(path_to_log_file)

    # Read log file
    with open(path_to_log_file, 'r') as f:
        content = f.readlines()

    # Create result dictionary
    res = {}

    # Create data frames
    for sep in separator_tuple:
        res[sep] = parse_to_frame(content, sep=sep, feature_processing_function=feature_processing_function)

    return res


def _extract_information(path_to_log_file_folder, content_tag, name_prefix, name_suffix, settings, create_node_pickle=False):
    """
    Creates a csv for each node in the log file.
    Each csv contains the features as observed by a node.
    :param path_to_log_file_folder:     str, the path to the folder containing the log file
    :param content_tag:                 str, tag describing data to extract from the log file
    :param name_prefix:                 str, prefix before the node id in the output file
    :param name_suffix:                 str, suffix after the name id in the output file, excluding '.csv'
    :return:
    """
    # Short cut for logger
    logger = settings.logger

    # Create path to file
    path_to_log_file = os.path.join(path_to_log_file_folder, "serial.log")

    # Check if file exists
    assert os.path.exists(path_to_log_file), "Path does not exist: {}".format(path_to_log_file)

    # Make output directory if necessary
    output_dir = os.path.join(path_to_log_file_folder, "individual")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Check for processing function
    if power_tag in content_tag:
        feature_processing_function = _process_powertracking_content
    else:
        feature_processing_function = _default_log_content_processing

    # Process log file to data frame
    data_set_dict = process_log_file(path_to_log_file, separator_tuple=(content_tag,),
                                     feature_processing_function=feature_processing_function)
    data_frame = data_set_dict[content_tag]

    # Process only valid data sets
    if data_frame.empty:
        logger.error("Data set seems to be empty. Make sure the pathes are correct and the log file is formatted correctly.\n" +
                     "Path: {}".format(path_to_log_file))
    elif not id_name in list(data_frame):
        logger.error("Data set does not contain node ids. Make sure the log file is formatted correctly.\n" +
                     "Path: {}".format(path_to_log_file))
    else:
        # Group data by node
        data_frame_grouped = data_frame.groupby(by=id_name)

        node_list = []

        # Iterate over groups
        for node_id in data_frame_grouped.groups:
            # Add to node list
            node_list.append(str(int(node_id)))

            # Get data of the group
            group_data = data_frame_grouped.get_group(node_id)

            # Id will be included in the name and not in the csv
            group_data = group_data.drop(labels=id_name, axis='columns')

            # Store data to csv
            group_data.to_csv(path_or_buf=os.path.join(output_dir, name_prefix + str(int(node_id)) + name_suffix + '.csv'),
                              index=False, sep='\t')

        # Pickle node ids
        if create_node_pickle:
            with open(os.path.join(output_dir, settings.nodes_dict_name), 'wb') as f:
                pickle.dump(node_list, f)
            logger.info("Saving mote dictionary to file {}".format(settings.nodes_dict_name))


def extract_feature_information(path_to_log_file_folder, settings):
    """
    Creates a csv for each node in the log file.
    Each csv contains the features as observed by a node.
    :param path_to_log_file_folder:            str, the path to the folder containing the log file
    :param settings:                        BatchConfig object, contains the current settings
    :return:
    """
    _extract_information(path_to_log_file_folder=path_to_log_file_folder, content_tag=feature_tag,
                         name_prefix='', name_suffix='',settings=settings, create_node_pickle=True)


def extract_power_information(path_to_log_file_folder, settings):
    """
    Extracts the power information for every node in the log file in path_to_log_file_folder.
    This information is then stored in a csv for each node.
    :param path_to_log_file_folder:         str, path to the folder containing the log file
    :param settings:                        BatchConfig object, contains the current settings
    :return:                                None
    """
    directories = []

    path = path_to_log_file_folder

    while True:
        path, directory = os.path.split(path)
        directories.append(directory)

        if directory == 'RPL-Attacks-Data' or directory is '':
            break

    directories.reverse()

    path_without = os.path.join(path, directories[0], 'without_AD', *directories[1:])

    _extract_information(path_to_log_file_folder=path_without, content_tag=power_tag,
                         name_prefix='power_', name_suffix='', settings=settings, create_node_pickle=True)

    _extract_information(path_to_log_file_folder=path_to_log_file_folder, content_tag=power_tag,
                         name_prefix='power_', name_suffix='', settings=settings, create_node_pickle=True)


def extract_score_information(path_to_log_file_folder, settings):
    """
    Extracts the scores from the log file in path_to_log_file_folder and stores it in a csv.
    :param path_to_log_file_folder:         str, path to the folder containing the log file
    :param settings:                        BatchConfig object, contains the current settings
    :return:
    """
    _extract_information(path_to_log_file_folder=path_to_log_file_folder, content_tag=score_tag,
                         name_prefix='scores_', name_suffix='', settings=settings)


def extract_edge_weights(path_to_log_file_folder, logger):
    """

    :param path_to_log_file_folder:
    :param logger:
    :return:
    """
    # Create path to file
    path_to_log_file = os.path.join(path_to_log_file_folder, "serial.log")

    # Check if file exists
    assert os.path.exists(path_to_log_file), "Path does not exist: {}".format(path_to_log_file)

    # Make output directory if necessary
    output_dir = os.path.join(path_to_log_file_folder, os.pardir, "results")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    logger.info("Extracting flow data from {}".format(path_to_log_file))

    # Get data frame from log file. This data frame contains id, from and to as observed by all nodes
    data_frame = process_log_file(path_to_log_file=path_to_log_file, separator_tuple=(flow_tag,),
                                  feature_processing_function=_process_flow_content)[flow_tag]

    # Group and count data
    assert not data_frame.empty, "Dataframe {} is empty!".format(path_to_log_file)
    package_count = data_frame.groupby(["From", "To"]).size().reset_index()

    return package_count
