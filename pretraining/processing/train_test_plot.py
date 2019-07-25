import os
import numpy as np
import pandas as pd
from pretraining.util.plotter import Plotter
from pretraining.util.helper import purge, _find_best_model, get_positives, load_threshold, _get_pickle


def train_test_plot(settings):
    """
    Train and predict
    :param settings:        The settings containing all the information
    :return:
    """
    settings.set_threshold(load_threshold(settings.global_pretrain_folder))

    # constants
    output_dir = os.path.join(os.path.dirname(settings.test_data_path), "individual")
    classification_img_name = "_classification.png"

    # get nodes in test set from pickle file, which is created in extract.py
    pickle_dir_path = os.path.join(os.getcwd(), output_dir)
    df_dictionary_keys = _get_pickle(settings=settings, path=pickle_dir_path)

    # purge both area and classification images
    settings.logger.info(
        "train_test_plot.py:\t\t\tPurging *{} and *{} from {}".format("_area.png", classification_img_name, output_dir))
    purge(logger=settings.logger, dir=output_dir, pattern="_area.png")
    purge(logger=settings.logger, dir=output_dir, pattern=classification_img_name)

    # create variable to store results
    score_batch = pd.DataFrame(columns=["score", "model", "features", "data"])

    # iterate over all mac adresses in the df_dict_keys, and perform classification and plotting
    for mac_adress in df_dictionary_keys:
        # test and test data path
        train_data_path = os.path.join(os.path.dirname(settings.training_data_path), "individual",
                                       mac_adress + ".csv")
        test_data_path = os.path.join(os.path.dirname(settings.test_data_path), "individual",
                                      mac_adress + ".csv")

        # run selected classifier
        r = _run(settings.model,
                 settings=settings,
                 train_data_path=train_data_path,
                 test_data_path=test_data_path,
                 output_folder=output_dir,
                 filename_stub=str(mac_adress))

        # ======================= END OF TRAIN AND TEST ========================
        # From here, it's only about scoring the result, e.g. determining how good it was
        val_test = r["res_main"]

        # assert that values are not NaN
        assert val_test is not None
        assert settings.model
        assert settings.features
        assert test_data_path

        # append to score batch
        data_dict = {"score": val_test, "model": str(settings.model), "features": str(settings.features),
                     "data": test_data_path}
        score_batch = score_batch.append(data_dict, ignore_index=True)

    # return to caller
    return score_batch


def _run(model, settings, output_folder=None, train_data_path=None, test_data_path=None, filename_stub=None):
    """
    Run a given model
    :param model:                   Instantiated Model, extending the BaseClassifier
    :param settings:                BatchConfig object
    :param output_folder:
    :param train_data_path:
    :param test_data_path:
    :param filename_stub:
    :return:
    """
    # Check input
    assert output_folder is not None, "Please set an apropiate output folder"

    # ============== Set Filenames ===================
    classific_results_plot_filename = filename_stub + "_py" + "_classification.png"
    df_plot_filename = filename_stub + "_area.png"

    # ============= Set test and train data path ================
    train_data_path = train_data_path if train_data_path else settings.training_data_path
    test_data_path = test_data_path if test_data_path else settings.test_data_path

    # =========================== Train ==========================
    # try:
    if not settings.use_pretrain_data:
        model.train_on_csv(train_data_path, settings)
    else:
        # we load a pretrained model, thus we do not need to train anything
        pass

    res = {}
    # ============================ Test =========================
    if settings.use_pretrain_data:
        # Read the data set
        data_set = pd.read_csv(test_data_path, sep='\t')
        grouped_data = data_set.groupby(by=settings.neighbors_feature)

        # Create dictionary for storing results
        res = {"df": data_set.drop(settings.neighbors_feature, axis='columns')}
        number_data_points = data_set.shape[0]
        res["res_main"] = np.zeros((number_data_points,))
        res["res_sup"] = np.zeros((number_data_points,))

        # Iterate based on the number of neighbors
        for number_neighbors in grouped_data.groups:
            # Find best classifer for each number of neighbors
            classifier_path = _find_best_model(number_neighbors=number_neighbors,
                                               pretrain_folder=settings.global_pretrain_folder)
            # Load the best classifier
            model.load_pretrained_model(classifier_path=classifier_path)

            # Get the data with number_neighbors neighbors
            group = grouped_data.get_group(number_neighbors).drop(settings.neighbors_feature, axis='columns')

            # Classify subset of data. Since we do not consider the sequence of data points, this does not yield problems
            temp_results = model.classify_on_data(df=group, settings=settings)

            # Transfer temp results to final results
            res["res_main"][group.index] = temp_results["res_main"]
            res["res_sup"][group.index] = temp_results["res_sup"]

    # ============================= Assert correct type of result ==========================================
    assert type(res["df"]) == pd.DataFrame and res["res_main"].size > 0
    # ============================= PUNISH NaN VALUES ============
    if np.isnan(res["res_main"]).any():
        print("PUNISHING THE CLASSIFIER FOR NAN VALUES!!!!!")
        res["res_main"][np.isnan(res["res_main"])] = -1e10

    if settings.save_images:
        positives = get_positives(res["res_main"], settings.threshold)

        # Plot the classification results (plots _py scores)
        Plotter.plot_results(res, settings,
                             output_folder=output_folder,
                             filename_for_saving=classific_results_plot_filename,
                             y_lim=settings.scores_y_lim,
                             color_red_above=settings.color_scores_red_above,
                             invert_coloring=settings.scores_invert_coloring,
                             additional_information_str=(
                                 train_data_path if train_data_path else settings.training_data_path,
                                 "Positives: {} ".format(positives))
                             )

        # Plot chart of the dataframe (plots AREA)
        Plotter.display_event_distribution_from_df(settings, res["df"], settings.logger,
                                                   output_folder=output_folder,
                                                   filename_for_saving=df_plot_filename,
                                                   stacked=settings.stacked,
                                                   )

        # Plot the anomaly score from the log file (plots _C scores)
        path = os.path.join(os.path.dirname(test_data_path), "scores_" + os.path.basename(train_data_path))
        try:
            data_frame = pd.read_csv(path, sep='\t')
            scores = data_frame["ANOMALY SCORE"]
            res = {"df": data_frame, "res_main": scores, "res_sup": scores}
            positives = get_positives(res["res_main"], settings.threshold)
            Plotter.plot_results(res, settings,
                                 output_folder=output_folder,
                                 filename_for_saving='' + classific_results_plot_filename.replace("py", "c"),
                                 y_lim=settings.scores_y_lim,
                                 color_red_above=settings.color_scores_red_above,
                                 invert_coloring=settings.scores_invert_coloring,
                                 additional_information_str=(
                                    train_data_path if train_data_path else settings.training_data_path,
                                    "Positives: {} ".format(positives))
                                 )
        except IOError as io_error:
            settings.logger.error(str(io_error) + "\nPlease run extract_scores first.")
    assert res is not None, "ERROR: Classifier returned {} a NONE object!".format(model)
    assert type(res) is dict, "ERROR: Classifier returned a non dictionary object. Please check code of {}".format(model)
    return res

