import os
import pickle
import shutil

import numpy
import numpy as np
import pandas
import multiprocessing
import copy

import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt

from pretraining.util.helper import collect_data


def generate_data_model(settings):
    """

    :param settings:
    :return:
    """
    # Dictionary for saving temporary results
    dic = collect_data(settings=settings, condition_function=lambda x: "without" in x and "control" not in x,
                       drop_first_n=settings.drop_first_n)
    # now, we have a dict containing for each number_of_neighbors the corresponding data from the nodes with this amount of neighbors
    # eg. dict[0] = [df1, df2, ...] contains all data from the nodes who have 0 neighbors
    max_neighbor = numpy.max(list(dic.keys()))

    # remove old folder
    if os.path.exists(settings.global_pretrain_folder):
        shutil.rmtree(settings.global_pretrain_folder)
    os.mkdir(settings.global_pretrain_folder)

    num_cores = multiprocessing.cpu_count()
    process_dict = {}

    # Create pretrained models for every model listed
    for model in settings.model_list:
        settings.logger.log(level=40, msg="Persisting model:\t\t{}".format(model))

        dict_list = _create_even_load_lists(data_dict=dic, number_lists=num_cores)
        for process_id, dictionary in enumerate(dict_list):
            model_copy = copy.copy(model)
            p = multiprocessing.Process(target=train_models, args=(dictionary, max_neighbor, model_copy, settings))
            process_dict[process_id] = p
            p.start()

        # Wait until all processes have finished
        for process_id in process_dict:
            process_dict[process_id].join()

    fig = plt.gcf()
    fig.set_size_inches(30, 30)
    plt.tight_layout()
    plt.savefig(os.path.join(settings.global_pretrain_folder, "historgrams_spline.png"), dpi=120)
    plt.close()


def train_models(dic, max_neighbor, model, settings):
    # Create a model for each number of neighbors
    for num_neighbors in dic.keys():

        # Concat all dataframes for a given number of neighbors into a single one
        data_frame = dic[num_neighbors]#pd.concat(dic[num_neighbors], axis=0)
        data_frame = data_frame.reset_index(drop=True)

        # Skip model training if data frame is empty
        if data_frame.shape[0] < 10:
            settings.logger.log(level=40, msg="Neighbors:\t{}\t\tToo small amount of data for training: {}".
                                format(int(num_neighbors), data_frame.shape[0]))
            continue

        # Logging, will only be reached if we have enough data
        settings.logger.log(level=40, msg="Neighbors:\t{}\t\tshape:{}".format(int(num_neighbors), data_frame.shape))

        # Train a model on the complete data (this takes a lot of time)
        model.train_on_data(data_frame, settings)
        settings.logger.log(level=40, msg="Training done for n={} with {} splines".format(num_neighbors,
                                                                                          settings.j_spline_intervals))

        # Plot a sample distribution
        for i, f in enumerate(settings.features):
            # Create a grid neighbors times features
            plt.subplot(len(settings.features), max_neighbor + 1, 1 + (i * (max_neighbor + 1)) + num_neighbors)

            # create x and y data (y data for both the KDE and the spline approximation)
            x = pandas.Series(np.linspace(start=0, stop=(data_frame[f].max() + 1) * 2, num=100))
            y_spline = model.get_model()["spline"][f].score_samples(x)
            y_kde = model.get_model()["kde"][f].score_samples(x.values.reshape(-1, 1))

            # plot log density
            plt.plot(x, y_spline, c='b', label="log spline")
            plt.plot(x, y_kde, c='r', label="log kde")

            # plot data distribution
            plt.plot(data_frame[f], [0.01] * len(data_frame[f]), '|', color='k')

            # Set x label and title
            plt.xlabel(str(f[:20]))  # , fontsize=5)
            plt.title("nn={}\n#={}".format(num_neighbors, data_frame[f].shape[0]))  # , fontsize=5)

        with open(os.path.join(os.getcwd(), settings.global_pretrain_folder, "{}.pkl".format(num_neighbors)),
                  'wb') as f:
            pickle.dump(model.get_model(), file=f, protocol=-1)


def _create_even_load_lists(data_dict, number_lists):
    """

    :param data_dict:
    :param number_lists:
    :return:
    """
    load_dict = [(key, data_dict[key].shape[0]**2) for key in data_dict]
    sorted_list = sorted(load_dict, key=lambda ele: -ele[1])

    list_of_dicts = [({}, 0) for i in range(number_lists)]

    for tuple in sorted_list:
        min_ele = min(list_of_dicts, key=lambda ele: ele[1])
        min_index = list_of_dicts.index(min_ele)
        list_of_dicts[min_index] = (list_of_dicts[min_index][0], tuple[1]+list_of_dicts[min_index][1])
        list_of_dicts[min_index][0][tuple[0]] = data_dict[tuple[0]]

    return [dictionary for (dictionary, value) in list_of_dicts]
