#include "feature-extraction.h"
#include "helper_funcs.h"

#include "net/ip/uip.h"
#include "net/ipv6/uip-ds6.h"
#include "net/ip/uip-debug.h"
#include "net/netstack.h"
#include "sys/ctimer.h"
#include "simple-udp.h"
#include "core/net/linkaddr.h"

#include "sys/rtimer.h"

#include <stdio.h>

/*---------------------------------------------------------------------------*/
/*  Sliding Window Count Feature Handling                                    */
/*---------------------------------------------------------------------------*/
static struct sliding_window_counts dio_counts, dis_counts, dao_counts;

swc_ptr get_data_ptr(feature_type f)
{
    switch(f)
    {
        case DIO_FEATURE: return &dio_counts;
        case DIS_FEATURE: return &dis_counts;
        case DAO_FEATURE: return &dao_counts;
        default: return 0;
    }
}


/*---------------------------------------------------------------------------*/
/*  Dodag Version Handling                                                   */
/*---------------------------------------------------------------------------*/
#define DODAG_VERSION_RESET_TIME 500
static struct ctimer dodag_version_reset_timer;
static unsigned dodag_version_count_bitmap[16];

/*---------------------------------------------------------------------------*/
/*  UDP SETTINGS                                                             */
/*---------------------------------------------------------------------------*/
#define MAX_NUM_NODES 14                // maximum number of neighbors we support. O(n^2) space required.
#define ROWS MAX_NUM_NODES              // See table below
#define COLS MAX_NUM_NODES              // ..
#define UDP_WINDOW_SIZE 500

static struct ctimer child_count_timer;
static unsigned udp_counts[ROWS][COLS] = {0};

/*---------------------------------------------------------------------------*/
/*  Neighbor Count Handling                                                  */
/*---------------------------------------------------------------------------*/
// TODO what if we have more neighbors than bits (16!) in variablel?!?!
static unsigned to_this_bitmap =  0;
static unsigned from_this_counts = 0;

static unsigned last_neighbor_count = 0;
static struct ctimer neighbor_count_timer;


void neighborhood_watch_handler(unsigned to_node, unsigned from_node, unsigned this_node)
{
    //rtimer_clock_t start, end;
    //start = RTIMER_NOW();
    // if send TO us, save FROM node
    if(this_node == to_node)
        to_this_bitmap |= 1u << from_node;

    // if send FROM us, send TO node
    if(this_node == from_node)
        from_this_counts |= 1u << to_node;

    //end = RTIMER_NOW();

    #if FEATURE_DEBUG
        printf("nbhwh: this=%d, from=%d, to=%d, int(to-this-bitmap)=%d, int(from-this-bitmap)=%d,\n",
                     this_node, from_node, to_node, to_this_bitmap, from_this_counts);
        printf("nbhwh: int(to-this-bitmap)="); print_bit_array(to_this_bitmap);
        printf("nbhwh: int(from-this-bitmap)="); print_bit_array(from_this_counts);
    #endif
    //printf("[Q] neighborhood_watch_handler: %lu\n", (unsigned long)end-start);
}


void update_neighbor_counts()
{
    // Compute number of neighbors and store in last_neighbor_count
    // TODO see above
    rtimer_clock_t start, end;
    start = RTIMER_NOW();
    unsigned neighbors = to_this_bitmap | from_this_counts;
    
    #if FEATURE_DEBUG
        printf("nbupdate: int(to-this-bitmap)=%d, int(from-this-bitmap)=%d, int(neigbor-bitmap)=%d, old_count=%d\n",
            to_this_bitmap, from_this_counts, neighbors, last_neighbor_count);
        printf("nbupdate: to-this-bitmap="); print_bit_array(to_this_bitmap);
        printf("nbupdate: from-this-bitmap="); print_bit_array(from_this_counts);
        printf("nbupdate: neighbors-bitmap="); print_bit_array(neighbors);
    #endif

    // Temporary count variable to prevent race condition when last_neighbor_count is queried
    unsigned last_neighbor_count_temp = 0;
    while(neighbors) {
        last_neighbor_count_temp += neighbors % 2;
        neighbors >>= 1;
    }

    last_neighbor_count = last_neighbor_count_temp;
    #if FEATURE_DEBUG
        printf("nbupdate: new_count=%d\n", last_neighbor_count);
    #endif

    // Reset package counters
    //to_this_bitmap =  0;
    //from_this_counts = 0;

    ctimer_reset(&neighbor_count_timer);
    end = RTIMER_NOW();
    unsigned long diff = 0;
    if (start > end){
        diff = (1<<16)-1 - start + end;
    }
    else {
        diff = end-start;
    }
    printf("[Q] update_neighbor_counts: %lu\n", diff);
}


unsigned get_num_neighbors()
{
    return last_neighbor_count;
}


void init_neighbor_count_loop()
{
     ctimer_set(&neighbor_count_timer, WINDOW_SIZE * CLOCK_SECOND, update_neighbor_counts, NULL);
}


/*---------------------------------------------------------------------------*/
/*  Sliding Window Count Feature Handling                                    */
/*---------------------------------------------------------------------------*/
void increment_packagecount(swc_ptr ptr)
{
    ptr->cnt_array[ptr->current_position]++;
}


void detector_packagecount_handler(feature_type f)
{
    increment_packagecount(get_data_ptr(f));
}


void slide_window_forward(swc_ptr ptr)
{
    /* RESET TIMER! */
    rtimer_clock_t start, end;
    start = RTIMER_NOW();
    ctimer_reset(&(ptr->timer));
    
    /* Slide forward */
    ptr->current_position = (ptr->current_position + 1) % WINDOW_SIZE;
    ptr->cnt_array[ptr->current_position] = 0;
    end = RTIMER_NOW();
    unsigned long diff = 0;
    if (start > end){
        diff = (1<<16)-1 - start + end;
    }
    else {
        diff = end-start;
    }
    printf("[Q] sliding_window_forward: %lu\n", diff);
}


unsigned sliding_window_sum(swc_ptr ptr) 
{
    //rtimer_clock_t start, end;
    //start = RTIMER_NOW();
    unsigned sum = 0;
    unsigned i;
    
    for (i = 0; i < WINDOW_SIZE; i++) 
       sum +=  ptr->cnt_array[i];

    //end = RTIMER_NOW();
    //printf("[Q] sliding_window_sum: %lu\n", (unsigned long)end-start);
    return sum;
}

unsigned get_feature_counts(feature_type f)
{
    return sliding_window_sum(get_data_ptr(f));
}


void init_sliding_window_loop(swc_ptr ptr) 
{
    // Initialize the sliding window
    ptr->current_position = 0;

    unsigned i;
    for (i = 0; i < WINDOW_SIZE; i++)
        ptr->cnt_array[i] = 0;

    /* SET TIMER! */
    ctimer_set(&(ptr->timer), CLOCK_SECOND, slide_window_forward, ptr);
}


void init_feature_loop(feature_type f)
{
    init_sliding_window_loop(get_data_ptr(f)); 
}


/*---------------------------------------------------------------------------*/
/*  Dodag Version Handling                                                   */
/*---------------------------------------------------------------------------*/
void call_detector_dodag_version_handler(uint16_t current_version) 
{
    int j = current_version / 16;
    int s = current_version % 16;
    dodag_version_count_bitmap[j] |= 1u << s;
    #ifdef FEATURE_DEBUG
    printf("Add Version-Number: %d. Add entry in array slot j = %d, s=%d\n", current_version, j, s);
    #endif
}

int get_dodag_version_difference(void)
{
    //rtimer_clock_t start, end;
    //start = RTIMER_NOW();
    unsigned dodag_version_count = 0;
    unsigned dodag_version_count_bitmap_copy;
    int j;
    for (j = 0; j < 16; j ++) {
        dodag_version_count_bitmap_copy = dodag_version_count_bitmap[j];
        while ( dodag_version_count_bitmap_copy) {
            dodag_version_count += dodag_version_count_bitmap_copy % 2;
            dodag_version_count_bitmap_copy >>= 1;
        }
    }
    //end = RTIMER_NOW();
    //printf("[Q] get_dodag_version_difference: %lu\n", (unsigned long)end-start);
    return dodag_version_count;
}

void reset_dodag_version_bitarray(void) {
    rtimer_clock_t start, end;
    start = RTIMER_NOW();
    int j;
    for (j = 0; j < 16; j ++) {
        dodag_version_count_bitmap[j] = 0;
    }
    ctimer_reset(&dodag_version_reset_timer);
    end = RTIMER_NOW();
    unsigned long diff = 0;
    if (start > end){
        diff = (1<<16)-1 - start + end;
    }
    else {
        diff = end-start;
    }
    printf("[Q] reset_dodag_version_bitarray: %lu\n", diff);
}

void init_dodag_version_loop(void)
{
    ctimer_set(&dodag_version_reset_timer, DODAG_VERSION_RESET_TIME * CLOCK_SECOND, reset_dodag_version_bitarray, NULL);
    int i;
    for (i = 0; i < 16; i++) {
        dodag_version_count_bitmap[i] = 0;
    }
}



/*---------------------------------------------------------------------------*/
/*  T/F Feature Handling                                                     */
/*---------------------------------------------------------------------------*/
void increment_udp_counts(unsigned to_node, unsigned from_node)
{
    //rtimer_clock_t start, end;
    //start = RTIMER_NOW();
    // check bounds
    if (from_node >= MAX_NUM_NODES || to_node >= MAX_NUM_NODES) {
        #if FEATURE_DEBUG
            printf("Internal UDP Table exceeded: Want to write to from_node=%d, to_node=%d when MAX_NUM_NODES=%d",
                from_node, to_node, MAX_NUM_NODES);
        #endif

        return;
    }

    printf("[UF] From=%d, To=%d;", from_node, to_node);
    #if FEATURE_DEBUG
    printf(" OldCount=%d -->", udp_counts[from_node][to_node]);
    #endif

    udp_counts[from_node][to_node] += 1;

    #if FEATURE_DEBUG
        printf(" From=%d, To=%d, NewCount=%d", from_node, to_node, udp_counts[from_node][to_node]);
    #endif
    printf("\n");
    //end = RTIMER_NOW();
    //printf("[Q] increment_udp_counts: %lu\n", (unsigned long)end-start);
}


// callback function to reset the counts window every 15 seconds
void increment_cc_array_position(void) 
{
    rtimer_clock_t start, end;
    start = RTIMER_NOW();
    // reset timer to retrieve the full result every second
    ctimer_reset(&child_count_timer);

    #if FEATURE_DEBUG
        printf("Resetting UDP counts...");
    #endif

    // Purge old entries
    int i, j;
    for (i = 0; i < ROWS; i += 1) 
        for (j = 0; j < COLS; j += 1) 
            udp_counts[i][j] = 0;
    end = RTIMER_NOW();
    unsigned long diff = 0;
    if (start > end){
        diff = (1<<16)-1 - start + end;
    }
    else {
        diff = end-start;
    }
    printf("[Q] increment_cc_array_position: %lu\n", diff);
}


// callback function to calculate the t/f value
/*---------------------------------------------------------------------------
   Number of Packets to Node per Second
UDP COUNTS

        TO_0    TO_1     TO_2   ...
FROM_0  12      0       21
FROM_1  120     331     228
FROM_2  0       0       0       <- possible Blackhole: Receives lots of packages, does not send any
....

*/
float get_feature_tf(void)
{
    //rtimer_clock_t start, end;
    //start = RTIMER_NOW();
    float max_val = 0;
    unsigned max_idx = 0;
    unsigned from_sums[ROWS] = {0}; // all packages sent FROM node at array index
    unsigned to_sums[COLS] = {0}; // all packges sent TO node at array index
    
    // Calculate row and column sums in one pass
    int i, j;
    for (i = 0; i < ROWS; i += 1) 
    {
        const unsigned* row_ptr = &udp_counts[i];
        for (j = 0; j < COLS; j += 1)
        {
            from_sums[i] += row_ptr[j];
            to_sums[j] += row_ptr[j];
        }
    }

    #if FEATURE_DEBUG
        print_multi_array(ROWS, COLS, udp_counts);
    #endif

    unsigned neighbors = to_this_bitmap | from_this_counts;

    #if FEATURE_DEBUG
        printf("Neighbors for t/f computation:"); print_bit_array(neighbors);
    #endif

    float tf;
    // Find maximum t/f quotient
    // ROWS = FROM
    // start at j=1, because j=0 is root and root cannot be anomalous note
    for (j = 1; j < ROWS; j += 1)
    { 
        // check if j-th bit in neighbors is set
        unsigned is_neighbor = neighbors & ( 1u << j);

        tf = ((float)to_sums[j] + 1) / ((float)from_sums[j] + 1);
        if (is_neighbor != 0 && tf > max_val)
        {
            max_val = tf;
            max_idx = j;
        }
    }

   
    #if FEATURE_DEBUG
        printf("From(Row) sums:");
        print_array(from_sums, ROWS);
        printf("To(Col) sums:");
        print_array(to_sums, COLS);
        printf("Max value %d at row %d.\n", (int)max_val, max_idx);
    #endif

    //end = RTIMER_NOW();
    //printf("[Q] get_feature_tf: %lu\n", (unsigned long)end-start);

    return max_val;
}

// main loop
void init_feature_t_f_count_loop(void) 
{
    // Init Code
    //printf("init_feature_t_f_count_loop: NETSTACK_RADIO.set_value(RADIO_PARAM_RX_MODE, 0)");
    
    /* Turn off RF frame filtering and H/W ACKs */
    int res = NETSTACK_RADIO.set_value(RADIO_PARAM_RX_MODE, 0);
    printf("Setting radio to promiscuous mode returned status code: %d. Radio_OK is %d\n", res, RADIO_RESULT_OK);

    //printf("//ctimer_set(&child_count_timer, UDP_WINDOW_SIZE * CLOCK_SECOND, increment_cc_array_position, NULL);");
    // timer to reset the counts table in which we store the number of to / from packets every 15 seconds
    ctimer_set(&child_count_timer, UDP_WINDOW_SIZE * CLOCK_SECOND, increment_cc_array_position, NULL);
}
