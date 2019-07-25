# -*- coding: utf8 -*-
from numpy import arccos, average, cos, pi, sign, sin, sqrt
from random import randint, shuffle, uniform
import copy

WSN_DENSITY_FACTOR = 3

__all__ = [
    'quadrants',
    'grid',
    'rectangle_const'
]


def draw_wsn(motes=None, algo='quadrants', **kwargs):
    """
    This function allows to draw the list of motes generated with one of the WSN generation functions hereafter.

    :param motes: a list of motes as output by one of the WSN generation functions hereafter
    """
    assert algo in __all__
    import networkx as nx
    from matplotlib import pyplot as plt
    motes = motes or eval(algo)(**kwargs)
    wsn = nx.Graph()
    for mote in motes:
        wsn.add_node(mote['id'])
    pos = {m['id']: (m['x'], m['y']) for m in motes}
    col = ['green'] + (len(motes) - 2) * ['blue'] + ['red']
    nx.draw(wsn, pos, node_color=col)
    plt.show()


def _malicious():
    """
    This function places the malicious mote in the middle of the network, not too close by the root.
    """
    global min_range, motes
    # add the malicious mote in the middle of the network
    # get the average of the squared x and y deltas
    avg_x = average([sign(m['x']) * m['x'] ** 2 for m in motes])
    x = sign(avg_x) * sqrt(abs(avg_x))
    avg_y = average([sign(m['y']) * m['y'] ** 2 for m in motes])
    y = sign(avg_y) * sqrt(abs(avg_y))
    # if malicious mote is too close by the root, just push it away
    radius = sqrt(x ** 2 + y ** 2)
    if radius < min_range:
        angle = arccos(x / radius)
        x, y = min_range * cos(angle), min_range * sin(angle)
    # TODO hacked the mote placements!
    # return {'id': len(motes), 'type': 'malicious', 'x': x, 'y': y, 'z': 0}
    return {'id': len(motes), 'type': 'malicious', 'x': -10, 'y': -10, 'z': 0}


# ************************************** NETWORK GENERATION FUNCTIONS ****************************************
"""
Each generation function generates a WSN with:
- 1 root (with ID 0)
- n legitimate motes
- 1 malicious mote (with ID n+1)

Given the following constraints:
- there should not by any isolated mote
- all motes must fit in to the squared area
"""


def quadrants(**kwargs):
    """
    This function generates positions according to quadrants in a range defined according to the aforementioned constraints.

    :return: the list of motes (formatted as dictionaries like hereafter)
    """
    global min_range, motes
    defaults = kwargs.pop('defaults')
    # TODO hacked to root mote placement: was x=y=0
    motes = [{'id': 0, 'type': "root", 'x': 20, 'y': -20, 'z': 0}]
    n = kwargs.pop('n', defaults["number-motes"])
    min_range = kwargs.pop('min_range', defaults["minimum-distance-from-root"])
    max_range = kwargs.pop('max_range', defaults["area-square-side"] // 2)
    tx_range = kwargs.pop('tx_range', defaults["transmission-range"])
    # determine 'i', the number of steps for the algorithm
    # at step i, the newtork must be filled with at most sum(f * 2 ** i)
    #   e.g. if f = 3, with 10 motes, root's proximity will hold 6 motes then the 4 ones remaining in the next ring
    i, s, ni = 1, 0, 0
    node_ids = list(range(1, n + 1))
    shuffle(node_ids)
    while s <= n:
        s += WSN_DENSITY_FACTOR * 2 ** i
        i += 1
    # now, generate the motes
    # first, the range increment is defined ; it will provide the interval of ranges for the quadrants
    range_inc = min(tx_range, max_range / (i - 1))
    for ns in range(1, i):
        # determine the number of motes to be generated inside the current ring
        n_step = min(WSN_DENSITY_FACTOR * 2 ** ns, n - ni)
        # determine the angle increment for the quadrants
        angle_inc = 360 // n_step
        # then, divide the ring in quadrants and generate 1 node per quadrant with a 10% margin either
        #  for the angle or for the range
        range_min, range_max = int((ns - 0.7) * range_inc), int((ns - 0.1) * range_inc)
        for j in range(0, n_step):
            ni += 1
            angle_min, angle_max = int((j + 0.25) * angle_inc), int((j + 0.75) * angle_inc)
            d, k, x, y = 0, 0, 0, 0
            while not min_range < d < tx_range * 0.9 and k < 1000:
                node_angle = randint(angle_min, angle_max) * pi / 180
                node_range = randint(max(range_min, min_range), min(range_max, max_range))
                # compute the coordinates and append the new node to the list
                x, y = node_range * cos(node_angle), node_range * sin(node_angle)
                for node in motes:
                    d = min(d, sqrt((x - node['x']) ** 2 + (y - node['y']) ** 2))
                k += 1
            motes.append({'id': node_ids[ni - 1], 'type': 'sensor', 'x': x, 'y': y, 'z': 0})
        if ni == n:
            break
        range_inc *= 0.75
    # finally, add the malicious mote in the middle of the network
    motes.append(_malicious())
    return sorted(motes, key=lambda o: o['id'])


def relocate_malicious_mote(logger, motes, experiments, current_sim):
    """
    Relocate malicious mote. Is expected to throw KeyError when "positions" not present.
    :param logger:
    :param motes:
    :param experiments:
    :param current_sim:
    :return:
    """
    # get settings
    new_mal_node_pos = experiments[current_sim]["positions"]["malicious"]
    num_of_malicious = int(experiments[current_sim]["num_malicious"])
    assert num_of_malicious < 5, "We support between 1 and 4 malicious nodes!"
    # remove mal nodes
    mal_node_template = motes[-1]
    del motes[-num_of_malicious:]
    assert mal_node_template["type"] == "malicious"
    mal_note_id_setoff = motes[-1]["id"] + 1
    for i in range(num_of_malicious):
        mal_node = copy.deepcopy(mal_node_template)
        mal_node["x"] = (1 if i < 2 else -1) * new_mal_node_pos["x"]
        mal_node["y"] = (1 if i % 2 == 0 else -1) * new_mal_node_pos["y"]
        mal_node["id"] = mal_note_id_setoff + i
        motes = motes + [mal_node]
    assert len([m for m in motes if m["type"] == "malicious"]) == num_of_malicious, \
        "Expecting {} malicious modes, but getting {}".format(num_of_malicious,
                                                                        ([m for m in motes if
                                                                             m["type"] == "malicious"]))
    assert len([m for m in motes if m["type"] == "root"]) == 1
    assert len(motes) > 2
    logger.debug(" > Replaced malicious motes position with {}".format(new_mal_node_pos))
    return motes


def grid(**kwargs):
    """
    This function generates positions according to a grid, defining square "layers" around the root mote.

    :return: the list of motes (formatted as dictionaries like hereafter)
    """
    global min_range, motes
    defaults = kwargs.pop('defaults')
    motes = [{'id': 0, 'type': "root", 'x': 0, 'y': 0, 'z': 0}]
    n = kwargs.pop('n', defaults["number-motes"])
    side = defaults["area-square-side"]
    min_range = kwargs.pop('min_range', defaults["minimum-distance-from-root"])
    max_range = 50 #kwargs.pop('max_range', side // 2)
    tx_range = kwargs.pop('tx_range', defaults["transmission-range"])
    # determine 'l', the number of layers for the algorithm
    l, s = 1, 0
    while s < n:
        s += 8 * l
        l += 1
    # now get the distance increment
    inc = min(side / (2 * l), .9*max_range)
    # then generate the positions
    node_id = 1
    for i in range(l):
        if node_id > n:
            break
        for j_x in [0, 1, -1]:
            if node_id > n:
                break
            for j_y in [0, 1, -1]:
                if node_id > n:
                    break
                x, y = i * inc * j_x, i * inc * j_y
                if x == y == 0:
                    continue
                x += uniform(-1, 1) * inc * .1
                y += uniform(-1, 1) * inc * .1
                motes.append({'id': node_id, 'type': "sensor", 'x': x, 'y': y, 'z': 0})
                node_id += 1
    # finally, add the malicious mote in the middle of the network
    motes.append(_malicious())
    return sorted(motes, key=lambda o: o['id'])


def rectangle_const(**kwargs):
    """
    This function places nodes like this:

    x   o   o   o

    o   o   o   r

    Where 'x' is malicious and 'r' is root.
    :param kwargs:
    :return:
    """
    motes = [{'id': 0, 'type': "root", 'x': -78, 'y': -21, 'z': 0},
             {'id': 1, 'type': "sensor", 'x': -42, 'y': -19, 'z': 0},
             {'id': 2, 'type': "sensor", 'x': -40, 'y': 20, 'z': 0},
             {'id': 3, 'type': "sensor", 'x': 0, 'y': -22, 'z': 0},
             {'id': 4, 'type': "sensor", 'x': 3, 'y': 24, 'z': 0},
             {'id': 5, 'type': "sensor", 'x': 41, 'y': -21, 'z': 0},
             {'id': 6, 'type': "sensor", 'x': 42, 'y': 18, 'z': 0},
             #{'id': 7, 'type': "sensor", 'x': 81, 'y': -22, 'z': 0},
             #{'id': 8, 'type': "sensor", 'x': 82, 'y': 19, 'z': 0},
             {'id': 7, 'type': "malicious", 'x': -3, 'y': 0, 'z': 0},
             ]

    return motes
