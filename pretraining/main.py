import matplotlib


matplotlib.use('Agg')

from pretraining.processing.generate_data_models import generate_data_model
from pretraining.processing.threshold import calculate_threshold
from pretraining.processing.statistics import get_function_time_statistics, get_size_statistics
from pretraining.processing.binomial_test import calculate_positive_probability
from pretraining.config.batch_config import BatchConfig
from pretraining.processing.train_test_plot import train_test_plot
from pretraining.processing.extract import extract_from_log_file, draw_udp_graph_wrapper
from pretraining.processing.super_batch import run_super_batch

import os

if __name__ == "__main__":
    settings = BatchConfig()

    # Extract from log files, if configured
    if settings.run_on_c_data:
        settings.logger.log(level=50, msg="\nSetting defaults for processing data sets based on C implementation...")
        settings.set_c_default()
        if settings.extract_features or settings.extract_power or settings.extract_scores:
            settings.logger.log(level=50, msg="\nExtracting data from log files...")
            run_super_batch(settings=settings, func=extract_from_log_file, multiprocess_features=0, multiprocess_datasets=1)

    # Draw UDP flow graphs
    if settings.draw_udp_graphs:
        settings.logger.log(level=50, msg="\nDrawing UDP flow graphs...:")
        run_super_batch(settings=settings, func=draw_udp_graph_wrapper, multiprocess_features=0, multiprocess_datasets=1)

    # iterate over all data sets and plot the data distribution per node
    if settings.run_generate_data_model:
        if settings.run_on_c_data:
            settings.set_c_default()
        settings.run_generate_data_model_default()
        settings.logger.log(level=50, msg="\nGenerating a mapping num_nodes => data distribution...:")
        generate_data_model(settings)

    if settings.run_calculate_threshold:
        if settings.run_on_c_data:
            settings.set_c_default()
        settings.run_generate_data_model_default()
        settings.logger.log(level=50, msg="\nCalculating threshold based on training data....")
        calculate_threshold(settings=settings)

    if settings.run_statistics:
        settings.set_c_default()
        settings.logger.log(level=50, msg="\nCalculating statistics....")
        calculate_positive_probability(settings=settings)

    if settings.run_power_statistics:
        settings.set_c_default()
        settings.logger.log(level=50, msg="\nCalculating power statistics....")
        get_function_time_statistics(settings=settings)

    if settings.run_size_statistics:
        settings.set_c_default()
        settings.logger.log(level=50, msg="\nCalculating size statistics....")
        get_size_statistics(settings=settings)

    # Configure the settings for train_test
    if settings.run_train_test_module:
        if settings.run_on_c_data:
            settings.set_c_default()
        # remove previous, intermediate file
        if os.path.isfile(settings.save_interm_results):
            os.remove(settings.save_interm_results)
        settings.logger.log(level=50, msg="\nPlotting the features and calculating Scores...:")
        settings.set_train_default()
        run_super_batch(settings, train_test_plot, multiprocess_features=0, multiprocess_datasets=not settings.plot_for_paper)


