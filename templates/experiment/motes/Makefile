SPLINE_DATA_FOLDER = ~/rpl-attacks/global_train

APPS += powertrace

$(info ===ANOMALY DETECTOR MOTE Build===)

#all: sensor-echo
all: sensor root malicious

# Remake spline-data if pkl files change
spline-data.h: $(wildcard $(SPLINE_DATA_FOLDER)/*.pkl)
	./make-spline-data.py $(SPLINE_DATA_FOLDER)


# Rebuild anomaly detection if spline-data was changed
anomaly-detection.c: spline-data.h
	touch anomaly-detection.c


CONTIKI={{ contiki }}
TARGET={{ target }}

#DEFINES = NETSTACK_CONF_RDC=contikimac_detector_driver
#PROJECT_SOURCEFILES += helper_funcs.c feature-extraction.c spline-detector.c package-parser.c detector-rdc.c anomaly-detection.c 

PROJECT_SOURCEFILES += helper_funcs.c feature-extraction.c spline-detector.c anomaly-detection.c package-parser.c

######## DECIDE WHAT FEATURES WE WANT TO ACTIVATE (COMMENT OUT WHAT YOU DONT WANT)
#CFLAGS += -DFEATURE_DEBUG
#CFLAGS += -DACTIVATE_POWERTRACE
CFLAGS += -DACTIVATE_AD

##### Features
CFLAGS += -DCONTIKI_WITH_IPV6=1
CFLAGS += -DUIP_CONF_IPV6=1
CFLAGS += -DUIP_IPV6_RPL=1 
#CFLAGS += -DUIP_CONF_UDP=1

##### Defines for Code Size Reduction
CFLAGS += -DSICSLOWPAN_CONF_FRAG=0
#CFLAGS += -DUIP_CONF_TCP=0
#CFLAGS += -DDEBUG=0
#CFLAGS += -DNDEBUG


##### the two following lines are aimed to reduce Contiki's size
CFLAGS += -ffunction-sections
LDFLAGS += -Wl,--gc-sections,--undefined=_reset_vector__,--undefined=InterruptVectors,--undefined=_copy_data_init__,--undefined=_clear_bss_init__,--undefined=_end_of_init__


include $(CONTIKI)/Makefile.include
