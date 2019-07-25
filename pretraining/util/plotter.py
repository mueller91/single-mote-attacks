import os
from time import strftime, localtime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from pretraining.util.helper import save_fig

plt.rcParams.update({'font.size': 42})


class Plotter:
    @staticmethod
    def _check_path_and_filename(path, filename):
        """
        Helper method, performs the follówing:
        1) create directory 'path' if not exists
        2) check if filename is valid filename (and not path)
        :param path:
        :param filename:
        :return:
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if "/" in filename or os.path.isabs(filename):
            raise ValueError('filename_for_saving has value ' + filename + ' which is not allowed.\n'
                                                                                      ' Must not be absolute path or contain "." or "/".')

    @staticmethod
    def display_event_distribution(settings, input_data_file, logger, output_folder, scale=True):
        """
        Create a 100% stacked plot of the input features, but exclude the TIME column if present.
        Saves image to disk. Wrapper for display_even_distribution_from_df
        :param input_data_file:         str
        :param logger:                  Logger
        :param output_folder:           Where to save it to
        :param scale:                   Will display percentage instead of absolute numbers
        :return:                        None
        """

        assert os.path.isfile(input_data_file)
        aggregated_df = pd.read_csv(input_data_file, sep='\t')
        filename_for_saving = ('full_' if scale else '') + os.path.basename(input_data_file)[:6]
        Plotter.display_event_distribution_from_df(settings, aggregated_df, logger, output_folder, filename_for_saving)

    @staticmethod
    def display_event_distribution_from_df(settings, aggregated_df, logger, output_folder, filename_for_saving,
                                           stacked=True):
        """
        Same as display_even_distribution, but takes DataFrame instead of path.
        :param filename_for_saving      str
        :param aggregated_df:           str
        :param logger:                  Logger
        :param output_folder:           Where to save it to
        :param scale:                   Will display percentage instead of absolute numbers
        :param ylim:
        :param stacked:
        :return:                        None
        """
        # set plot size
        plot_size =42
        
        # check that aggregated_df really is of type DF
        assert isinstance(aggregated_df, pd.DataFrame), "Data passed to plotter must be of type DataFrame!"

        # For the event distribution plot, select every column except the total-time column
        assert not aggregated_df.empty, "Dataframe must not be empty"

        if stacked and (aggregated_df < 0).any(axis=0).any():
            logger.warn("Plotter: =====================================================================\n"
                        "\t\tCareful! Plotting in stacked mode, but some columns contain negative values.\n"
                        "\t\tShifting these upwards. May render comparision with other features inconsistent.\n"
                        "\t\tNegative columns are:\n{}\n".format((aggregated_df < 0).any(axis=0)))
            # we may encounter negative values in test set, even though we run MinMax in Classifier.
            # subtract negative minima from feature, do not touch features without negative values
            neg_filter = aggregated_df.min()
            neg_filter[neg_filter >= 0] = 0
            aggregated_df = aggregated_df - neg_filter

        # Scale the plot, so the sum of all features is 1
        if settings.scale:
            values = aggregated_df.values
            scaled_values = normalize(values, norm="l1")
            aggregated_df = pd.DataFrame(scaled_values, columns=aggregated_df.columns)
            settings.plot_ylim = (-0, 1)

        ax = aggregated_df.plot(kind='area',
                                stacked=stacked,
                                title=(str(os.path.join(output_folder, filename_for_saving)[-100::])
                                + "\n" + strftime("%Y-%m-%d %H:%M:%S", localtime())) if not settings.plot_for_paper
                                else "",
                                # title="Area chart of the SDN controller's services",
                                ylim=settings.plot_ylim)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(plot_size)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(plot_size)

        # set labels
        if not settings.plot_for_paper:
            ax.set_ylabel("instances"
                          + (", normalized" if settings.normalize else "")
                          + (", MinMax scaled" if settings.minmax_scaling else "")
                          + (", scaled" if settings.scale else "")
                          + (", PCA dim n = {}".format(settings.n_dim_reduction) if settings.n_dim_reduction else "")
                          + (", pretrained" if settings.use_pretrain_data else "")
                          + (", spline-approximation" if settings.use_spline_fitting else ""),
                          fontsize=plot_size)
            ax.set_xlabel('data' + ", first {} data points dropped".format(settings.drop_first_n)
                          if settings.drop_first_n > 0 else "", fontsize=plot_size)
        else:
            ax.set_ylabel("feature value", fontsize=plot_size)
            ax.set_xlabel("data", fontsize=plot_size)

        # Set margins to avoid "whitespace"
        ax.margins(0, 0)

        # Put a legend to the right of the current axis
        ax.legend(loc='upper center', prop={'size': plot_size})

        # Check if output folder exists, if not create it
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # First, check consistency in file and path variables. Then, save to output folder and log that
        Plotter._check_path_and_filename(output_folder, filename_for_saving)
        save_to = os.path.join(output_folder, filename_for_saving)
        save_fig(plt.gcf(), save_to, dpi=100)
        logger.info('Plotter:\t\t\t\t\tSaving "area_chart" to ' + save_to
                    + ' using Scale: ' + str(settings.scale) + ', Stacked: ' + str(stacked))
        plt.close()

    @staticmethod
    def plot_results(res_dict, settings, output_folder, filename_for_saving, y_label='',
                     invert_coloring=False, y_lim=None, color_red_above=1.0,
                     additional_information_str=None):
        """
        Plot results of classifier
        :param res_dict:                    dictinaory with keys df, res_main, res_sup
        :param settings:
        :param additional_information_str:  str, additional information to print on plot for later reference
        :param color_red_above:             float, threshold above which to color points red
        :param invert_coloring              bool, swap red and blue if true
        :param output_folder:               str
        :param filename_for_saving:         Save image to output_folder/filename (.png, automatically added if not present)
        :param y_label:                     str, label for y axis
        :param y_lim:                       tuple, optional. if present, set y_lim[lower, upper]
        :return:                            None, but has side effects: Writes plot to disk
        """
        # set plot size
        plot_size = 62

        # extract from results dict
        r = res_dict["res_main"]
        sup = res_dict["res_sup"]

        assert len(r) > 0
        r = np.copy(r)
        if y_lim:
            # replace values on bottom or top if needed
            # for res_main
            r[r < y_lim[0]] = y_lim[0]
            r[r > y_lim[1]] = y_lim[1]

            # for sup results
            sup[sup < y_lim[0]] = y_lim[0]
            sup[sup > y_lim[1]] = y_lim[1]
        r = r.astype(float)

        # TODO SHIFT x axis by offset!
        x = np.arange(start=settings.drop_first_n, stop=settings.drop_first_n + len(r), step=1)

        # apply blue and red coloring
        outlier = r.copy()
        normal = r.copy()
        outlier[outlier < color_red_above] = np.nan
        normal[normal >= color_red_above] = np.nan
        plt.plot(x, outlier, '.', c='r' if not invert_coloring else 'b', markersize=49)
        plt.plot(x, normal, '.', c='b' if not invert_coloring else 'r', markersize=49)

        # plot suppl.
        plt.plot(x, sup, 'x', c='purple', markersize=39)


        # xlim, title and optionally ylim
        plt.xlim(settings.drop_first_n, r.size + settings.drop_first_n)
        if not settings.plot_for_paper:
            plt.title(os.path.join(output_folder, filename_for_saving)
                  + "\n" + strftime("%Y-%m-%d %H:%M:%S", localtime())
                  + ", ".join([str(x) for x in additional_information_str if x is not None]))
        if y_lim:
            diameter = abs(y_lim[0] - y_lim[1])
            plt.ylim(y_lim[0] - diameter * 0.1, y_lim[1] + diameter * 0.1)
        plt.ylabel("score", fontsize=plot_size)
        plt.xlabel("data", fontsize=plot_size)

        ax = plt.gca()
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(plot_size)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(plot_size)

        plt.gcf().subplots_adjust(bottom=0.30)

        # =================================================================
        # save plot
        Plotter._check_path_and_filename(path=output_folder, filename=filename_for_saving)
        save_to = os.path.join(output_folder, filename_for_saving)
        save_fig(plt.gcf(), save_to, dpi=100, width=30, height=10)
        settings.logger.info("Plotter:\t\t\t\t\tSaving classification to " + save_to)

        plt.close()
