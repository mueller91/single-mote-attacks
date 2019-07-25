import os

from pretraining.processing.log_parser import extract_power_information, extract_feature_information, extract_score_information
from pretraining.util.visualisation import draw_udp_flow_graph


def extract_from_log_file(settings):
    """
    Extracts data from log files. The paths to the data sets and the folders containing the log files has to be
    specified in settings.
    :param settings:        BatchConfig object, an object containing all relevant information for feature extraction.
    :return:                None
    """
    # Shortcut for logger
    logger = settings.logger

    logger.info("extract.py:\t\t\t\t\t\tPreprocessing IOT data, extracting from log files: "
                "feature data: {} power data: {}".format(settings.extract_features, settings.extract_power))

    # Extract data from log files
    path_to_data_set = os.path.dirname(settings.test_data_path)
    if settings.extract_power:
        extract_power_information(path_to_log_file_folder=path_to_data_set, settings=settings)
    if settings.extract_features:
        extract_feature_information(path_to_log_file_folder=path_to_data_set, settings=settings)
    if settings.extract_scores:
        extract_score_information(path_to_log_file_folder=path_to_data_set, settings=settings)


def draw_udp_graph_wrapper(settings):
    """

    :param settings:
    :return:
    """
    draw_udp_flow_graph(path=os.path.join(os.path.dirname(settings.test_data_path), os.pardir), logger=settings.logger,
                        use_log_files=settings.run_on_c_data)

