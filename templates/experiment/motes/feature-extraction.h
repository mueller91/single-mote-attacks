#ifndef FEATURE_EXTRACTION_H_
#define FEATURE_EXTRACTION_H_

#include "sys/ctimer.h"

/*---------------------------------------------------------------------------*/
/*  Data type definitions                                                    */
/*---------------------------------------------------------------------------*/
#define WINDOW_SIZE 15

typedef enum { DAO_FEATURE=0, DIO_FEATURE=1, DIS_FEATURE=2, UDP_TF_FEATURE=3, DODAG_VERSION_FEATURE=4} feature_type;

struct sliding_window_counts {
    feature_type f_type;
    unsigned window_size;
    unsigned current_position;
    unsigned cnt_array[WINDOW_SIZE];
    struct ctimer timer;
    };

typedef struct sliding_window_counts* swc_ptr;


/*---------------------------------------------------------------------------*/
/*  Neighbor Count Handling                                                  */
/*---------------------------------------------------------------------------*/
void neighborhood_watch_handler(unsigned to_node, unsigned from_node, unsigned this_node);

unsigned get_num_neighbors();

void init_neighbor_count_loop();


/*---------------------------------------------------------------------------*/
/*  Sliding Window Count Feature Handling                                    */
/*---------------------------------------------------------------------------*/
void detector_packagecount_handler(feature_type f);

unsigned get_feature_counts(feature_type f);

void init_feature_loop(feature_type f);


/*---------------------------------------------------------------------------*/
/*  Dodag Version Handling                                                   */
/*---------------------------------------------------------------------------*/
void call_detector_dodag_version_handler(uint16_t);

void init_dodag_version_loop(void);

int get_dodag_version_difference(void);


/*---------------------------------------------------------------------------*/
/*  T/F Feature Handling                                                     */
/*---------------------------------------------------------------------------*/
void increment_udp_counts(unsigned to_node, unsigned from_node);

void init_feature_t_f_count_loop(void);

float get_feature_tf(void);

#endif /* FEATURE_EXTRACTION_H_ */