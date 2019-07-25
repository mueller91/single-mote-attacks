from subprocess import call

from os.path import expanduser, join

user = expanduser("~/")
virtual_env = "contiki_env"

RPL_ATT_CONF = join(user, ".rpl-attacks.conf")

# Define the directories of the experiments
EXPERIMENTS = [
        "1l", "2l", "3l", "4l",
        "5l", "6l", "7l", "8l",
        "9l", "10l", "11l", "12l",
        "13l", "14l", "15l", "16l",
        "17l", "18l", "19l", "20l"
        ]

for exp in EXPERIMENTS:
    # change conf
    with open(RPL_ATT_CONF, 'w') as f:
        msg = "[RPL Attacks Framework Configuration]\n" +\
        "contiki_folder = " + join(user, "contiki") + \
        "\nexperiments_folder = " + join(user, "RPL-Attack-Data/{}/\n"
                                           .format(exp))
        f.write(msg)
    call(join(user, virtual_env, "bin/fab make_all:{}".format(exp)), shell=True)
    call(join(user, virtual_env, "bin/fab run_all:{}".format(exp)), shell=True)
