import os
import seaborn as sns
from re import finditer, MULTILINE
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from pretraining.processing.log_parser import extract_edge_weights


def get_motes_from_simulation(simfile, as_dictionary=True):
    """
    This function retrieves motes data from a simulation file (.csc).

    :param simfile: path to the simulation file
    :param as_dictionary: flag to indicate that the output has to be formatted as a dictionary
    :return: the list of motes formatted as dictionaries with 'id', 'x', 'y' and 'motetype_identifier' keys if
              short is False or a dictionary with each mote id as the key and its tuple (x, y) as the value
    """
    motes = []
    with open(simfile) as f:
        content = f.read()
    iterables, fields = [], ['mote_id']
    for it in ['id', 'x', 'y', 'motetype_identifier']:
        iterables.append(finditer(r'^\s*<{0}>(?P<{0}>.*)</{0}>\s*$'.format(it), content, MULTILINE))
    for matches in zip(*iterables):
        mote = {}
        for m in matches:
            mote.update(m.groupdict())
        motes.append(mote)
    if as_dictionary:
        motes = {int(m['id']): (float(m['x']), float(m['y'])) for m in motes}
    return motes

def parse_udp_flows(pcap_csv_file):
    """
    This function filters the pcap file for UDP packages and selects only the
    root-bound traffic which is aggregated to a per-edge count

    :param pcap_csv_file: path to the pcap-csv-logfile
    """
    df = pd.read_csv(pcap_csv_file, sep='\t', dtype=str)

    mac_src_host_selector = "_source.layers.wpan.wpan.src64"
    mac_dst_host_selector = "_source.layers.wpan.wpan.dst64"
    ip_dest_selector = '_source.layers.ipv6.ipv6.dst'

    root_ip_address = "aaaa::c30c:0:0:0"

    udp_traffic = df.loc[df["_source.layers.udp.udp.length"].notnull()]
    udp_traffic = udp_traffic[[mac_src_host_selector, mac_dst_host_selector, ip_dest_selector]]
    del df

    root_bound_traffic = udp_traffic.loc[udp_traffic[ip_dest_selector] == root_ip_address, [mac_src_host_selector, mac_dst_host_selector]]
    root_bound_traffic = root_bound_traffic.applymap(lambda x: int(str(x).split(":")[-1], 16))
    package_counts = root_bound_traffic.groupby([mac_src_host_selector, mac_dst_host_selector]).size().reset_index()

    return package_counts


def draw_udp_flow_graph(path, logger, use_log_files=False):
    """
    This function plots the root-bound UDP package for each edge. The edges are labeled with the corresponding
    package count with high-flow edges visualized thicker than low-flow edges.

    :param path: path to the pcap-csv-logfile (including [with-|without-malicious])
    """
    plt.clf()
    with_malicious = (os.path.basename(os.path.normpath(path)) == 'with-malicious')
    data, results = os.path.join(path, 'data'), os.path.join(path, 'results')

    try:
        if not use_log_files:
            edge_weights = parse_udp_flows(os.path.join(data, 'output.csv'))
        else:
            edge_weights = extract_edge_weights(data, logger)
            pass
    except Exception as e:
        logger.error("Error parsing pcap file: %s!" % e.message)
        return

    G = nx.DiGraph()

    pos = get_motes_from_simulation(os.path.join(path, 'simulation.csc'))
    G.add_nodes_from(pos.keys())

    for n, p in pos.items():
        x, y = p
        G.node[n]['pos'] = pos[n] = (x, -y)

    for e in edge_weights.values:
        G.add_edge(e[0], e[1], weight=e[2], label=str(e[2]))

    colors = G.number_of_nodes() * ['y']
    colors[0] = 'g'
    if with_malicious:
        colors[-1] = 'r'

    to_root_edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    to_root_edge_weights = np.array([v for v in to_root_edge_labels.values()])

    plt.figure(figsize=(10, 10))

    nx.draw(G, pos,
            node_color=colors,
            width=0,
            node_size=3000,
            font_size=30)

    nx.draw_networkx_labels(G, pos,
                            font_size=30,
                            font_weight='bold',
                            font_family='sans-serif'
                            )

    def _scaling(values, a, b):
        return ((values * 1.0) / values.max()) * (b - a) + a

    nx.draw_networkx_edges(G, pos,
                           edgelist=to_root_edge_labels.keys(),
                           alpha=0.85,
                           edge_color='b',
                           # edge_color=toRootEdgeWeights,
                           # edge_cmap=plt.cm.gnuplot2,
                           width=_scaling(to_root_edge_weights, 4.0, 15.0),
                           arrows=True,
                           arrowsize=30,
                           arrowstyle='->',
                           font_size=100
                           )

    plt.savefig(os.path.join(results, "udp_flow_graph.png"), arrow_style=FancyArrowPatch)


def _show_data(data, name):

    feature = 'CPU Time'

    # Prepare data frame for plotting
    plotdf = pd.melt(data.loc[:, ["ID", feature]], id_vars=["ID"], var_name=feature, value_name='Time')

    # Create grid plot
    grid = sns.FacetGrid(plotdf, col="ID", row=feature, hue=feature, margin_titles=True)
    grid.map(plt.plot, "Time", marker="o", ms=4)
    grid.fig.tight_layout(w_pad=1)

    # Show plot and save to directory
    plt.savefig(os.path.join("rpl-attacks/global_train", name + ".png"))
    plt.show()
