import numpy as np
import os

from pretraining.util.helper import collect_data, _find_best_model


def calculate_threshold(settings):
    """

    :param settings:
    :return:
    """
    # Dictionary for saving
    dic = collect_data(settings=settings, condition_function=lambda x: "without" in x and "control" not in x)

    results = None

    for model in settings.model_list:
        # Create a model for each number of neighbors
        for num_neighbors in dic.keys():

            if num_neighbors != 0:

                # Concat all dataframes for a given number of neighbors into a single one
                data_frame = dic[num_neighbors]#pd.concat(dic[num_neighbors], axis=0)
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

    threshold = np.percentile(results, settings.train_percentile)
    for i in range(100):
        if (i % 25 ) == 0:
            print("{}-Percentile: {}".format(i, np.percentile(results, i)))
    settings.set_threshold(threshold=threshold)

    # Save threshold to file
    with open(os.path.join(settings.global_pretrain_folder, "threshold.txt"), 'w') as file:
        file.write("threshold: {}".format(threshold))
