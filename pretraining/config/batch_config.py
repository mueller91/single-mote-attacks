import numpy as np
from random import shuffle
from pretraining.classifier.kernel_density_classifier import KernelDensityClassifier
from pretraining.util.helper import create_logger


class BatchConfig:
    def __init__(self):

        # ============================== DEFINE MODULES TO BE RUN ===================================
        # Define what to run

        # Run on data from the C implementation if true, from the PCAPs otherwise.
        # If it is true and either run_parse_module or run_extract_module is set, we extract data from the log files.
        # This data is used for training.
        self.run_on_c_data =                bool(1)

        self.extract_features =             bool(1)

        self.extract_power =                bool(1)

        self.extract_scores =               bool(1)

        self.draw_udp_graphs =              bool(1)

        # Aggregate all available data into several big dataframes, sorted by number of neighbors for global pretraining
        self.run_generate_data_model =      bool(1)

        # Calculate a threshold based on a percentile of the classification of the trainings data
        self.run_calculate_threshold =      bool(1)

        self.run_statistics =               bool(1)
        self.run_power_statistics =         bool(1)
        self.run_size_statistics =          bool(1)

        # Run the train and test module with either the global pretrain data, or on the clean control set
        # Plots area charts
        self.run_train_test_module =        bool(1)

        # ============================== Define model threshold ======================================
        self.threshold = -23.88024253715127
        self.train_percentile = 1

        # ============================== Image related ===============================================
        # Define if images (of the features and scores) should be saved
        self.save_images = bool(1)

        # Defaults to 0. If 1, create a power set of the features and iterate through it. Useful only for feature selection.
        self.create_power_set = bool(0)

        # =============================== Data paths for config ======================================
        # Paths to the data sets. Comment in and out as needed!
        self.data_sets = [
            "../../RPL-Attacks-Data/1l",
            "../../RPL-Attacks-Data/2l",
            "../../RPL-Attacks-Data/3l",
            "../../RPL-Attacks-Data/4l",
            "../../RPL-Attacks-Data/5l",
            "../../RPL-Attacks-Data/6l",
            "../../RPL-Attacks-Data/7l",
            "../../RPL-Attacks-Data/8l",
            "../../RPL-Attacks-Data/9l",
            "../../RPL-Attacks-Data/10l",
            "../../RPL-Attacks-Data/11l",
            "../../RPL-Attacks-Data/12l",
            "../../RPL-Attacks-Data/13l",
            "../../RPL-Attacks-Data/14l",
            "../../RPL-Attacks-Data/15l",
            "../../RPL-Attacks-Data/16l",
            "../../RPL-Attacks-Data/17l",
            "../../RPL-Attacks-Data/18l",
            "../../RPL-Attacks-Data/19l",
            "../../RPL-Attacks-Data/20l",
        ]

        # Paths to test data files. Comment in and out as needed!
        self.relative_test_data_paths = [
            "control-set-clean/without-malicious/data",
            "blackhole-attack/with-malicious/data",
            "blackhole-attack/without-malicious/data",
            "hello-flood-attack/with-malicious/data",
            "hello-flood-attack/without-malicious/data",
            "version-number-attack/with-malicious/data",
            "version-number-attack/without-malicious/data",
        ]

        # ==================================== MAIN Features =======================================
        # set the features with which to work
        self.features = [
            # ============ FEATURES USED IN THE PAPER ========
            'DIS sum 15s',
            'DIO sum 15s',
            'DAO sum 15s',
            'Nr of DODAG Versions',
        ]

        # ============================= ADDITIONAL FEATURES ==========================
        # These work in conjunction with the for loop below.
        self.area_feature_timedeltas = [
            50
        ]

        # Set the step size of the sliding window
        self.step_size_fraction = 0.5

        # Select which area features are to be used
        for t in self.area_feature_timedeltas:
            self.features.extend([
                "udp t/f " + str(t),
            ])

        # Name of the feature containing the number of neighbors
        self.neighbors_feature = "number_neighbors"

        # Drop first n entries in order to avoid irregularities during DODAG construction. n in seconds
        self.drop_first_n = 0

        self.drop_first_n_for_positives = 000

        # ========================================== Data paths ========================================
        # Default name of the test data file
        self.test_file_name = "output.csv"

        # Relative path to the training data file within a data set
        self.relative_training_data_path = "control-set-clean/without-malicious/data/output.csv"

        # name of the nodes pickle file
        self.nodes_dict_name = "nodes_list.pickle"

        # where to save pretrain files
        self.global_pretrain_folder = "../global_train/"

        # ================================= Data paths for running =======================================
        # These are overwritten and serve only as temporary storage when running several calculations in sequence
        # Data path to the current data set. Defaults to the x data set.
        self.current_data_set_path = ""

        # Path to the training data file including the current data set path
        self.training_data_path = ""

        # Path to current test data
        self.test_data_path = ""

        # ==================================== Spline fitting ===================================
        # use spline fitting?
        self.use_spline_fitting = bool(1)

        self.j_spline_intervals = 5

        # ================================== Data paths for model evaluation =============================
        self.score_results_file_final = "results/out.csv"
        self.save_interm_results = "results/intermediate_results.csv"
        self.folder_for_parial_interm_results = "results/partial_interm"

        # Default value for the feature combinations. The default is only "using all features".
        self.feature_combinations = [self.features]

        # Number of subsets from the power set (if larger than set size, take whole set)
        self.powerset_subset_size = 5 * 1e1

        # Initialize feature_to_number_dict
        self.features_to_number_dict = {}
        self._create_feature_to_number_dict()

        # ==================================== Logger =============================================
        # set logger and log level
        self.logger = create_logger(40)         # The level of the logger to use throughout the project. 20 shows all, 40 only an overview

        # ===================================== Plotting information =================================
        # Plotting options for scores
        self.scores_invert_coloring = bool(1)   # invert the standart coloring: If true, color red below the threshold and blue above
        self.color_scores_red_above = -1500       # Threshold above which to color red, and below which to color blue
        self.scores_y_lim = (-3*10**1, 1)           # Limit the y axis of the scores plot

        # Options for Area Plotting
        self.scale = bool(0)                    # Scale all features to percentages by dividing by the sum of each observation
        self.stacked = bool(0)                  # Plot features stacked on top of each other
        self.plot_ylim = None                   # (-0, 5000)  # For Area Plot: plot featues between these values

        # Configure axis labeling depending on whether we plot the figure for a paper
        self.plot_for_paper = bool(0)
        self.set_threshold(threshold=self.threshold)

        # ======================================= Generator data =========================================
        # In order to be able to adapt the features of the models, we need to use a generator
        # By default these are all empty and HAVE to be set with e.g. set_train_default()
        self.generator_hmm_n = []
        self.generator_hmm_emis_dist = []
        self.generator_hmm_epochs = []
        self.generator_hmm_seq_length = []
        self.generator_kd_n = []
        self.generator_gauss_seq_length = []
        self.generator_svm_kernel_list = []

        # ====================================== Model information ======================================
        # Options for scores    (are used also in area plotting)
        self.dim_reduction = bool(0)
        self.normalize = bool(0)  # normalise using euclidean distance, e.g. L2-norm
        self.minmax_scaling = bool(0)  # do MinMax scaling, which scales each feature individually to [0, 1]
        self.n_dim_reduction = None  # n_components to reduce to when doing dimensionality reduction
        self.epochs = 100  # number of epochs for training
        self.sequence_length = 10  # Length of a single sequence when using sequence based methods

        # The model which is used by train_test_plot
        self.model = None

        # The list of models for iterating in super batches
        self.model_list = self.model_generator()

    def model_generator(self):
        """
        A simple generator for testing if the system runs. Can be used like a iterable.
        :return:                    The next model in the list.
        """
        # Kernel Density
        yield KernelDensityClassifier(self.logger, pca_components=self.n_dim_reduction, independand_bandwidth=True,
                                      features_to_select=self.features, atol=0, rtol=1e-8,
                                      use_spline_fitting=self.use_spline_fitting)

    def run_generate_data_model_default(self):
        self.set_default()

    def set_default(self):
        # Set generator parameters to define some classifiers classifier
        self.generator_hmm_n = []  # [5]
        self.generator_hmm_emis_dist = []  # ["normal"]
        self.generator_hmm_epochs = []  # [100]
        self.generator_hmm_seq_length = []  # [20]
        self.generator_kd_n = [5]
        self.generator_gauss_seq_length = []  # self.generator_hmm_seq_length
        self.generator_svm_kernel_list = []  # ["rbf"]

        # Create feature_to_number_dict
        self._create_feature_to_number_dict()

        # Set the name of the test file to be the csv from which we want to extract from
        self.test_file_name = "output.csv"

        # Reset the model generator
        self.model_list = self.model_generator()

        # Set feature combination list to only contain one combination
        self.feature_combinations = [self.features]

    def set_c_default(self):
        self.set_default()
        self.nodes_dict_name = "c_nodes_list.pickle"
        self.features = [
            "DIS",
            "DIO",
            "DAO",
            "Vers-Nums",
            "T_F"
        ]
        self.neighbors_feature = "NEIGHBORS"

    def set_train_default(self):
        # Set feature combination list to only contain one combination
        if self.create_power_set:
            self.feature_combinations = self._create_random_feature_combinations(self.powerset_subset_size)

        self.set_default()

    def set_threshold(self, threshold):
        """
        Sets the threshold and plotting limits accordingly.
        :param threshold:       float, threshold to set
        :return:
        """
        self.threshold = threshold
        self.color_scores_red_above = threshold
        if threshold < 0:
            self.scores_y_lim = (2 * threshold, 1)
        else:
            self.scores_y_lim = (0.5 * threshold, 2 * threshold)

    def _create_random_feature_combinations(self, size):
        """
        Create some random feature combination for evaluation.
        :param size:            int, The number of different feature combinations
        :return:                list of lists, A list containing all calculated feature combinations
        """
        feature_list = self.features
        assert len(feature_list) > 0, "The feature list is empty. Please define at least one feature."
        assert size > 0, "Chose at least one set of features!"
        return _random_subset(_power_set(feature_list), size)

    def _create_feature_to_number_dict(self):
        """
        Creates a dictionary mapping features to numbers. It will be saved in the variable 'self.feature_to_number_dict'.
        :return:        None
        """
        feature_list = self.features
        self.feature_to_number_dict = {}
        for i in range(len(feature_list)):
            self.features_to_number_dict[feature_list[i]] = str(i)


def _power_set(set):
    """
    Construct the power set to a given list.
    :param set:         list, Base set for the power set
    :return:            list of lists, The list contains all subsets of set
    """
    power_set = [[]]
    for element in set:
        # Iterate over subsets
        for subset in power_set:
            # Add new element
            power_set = power_set + [list(subset) + [element]]
    power_set.remove([])
    return power_set


def _random_subset(set, size):
    """
    Choose a random subset of elements of a given subset. The elements are chosen without replacement.
    :param set:         list, The list to choose from
    :param size:        int, Number of examples drawn from the set
    :return:
    """
    if size < len(set):
        return np.random.choice(set, size=int(size), replace=False)
    else:
        shuffle(set)
        return set
