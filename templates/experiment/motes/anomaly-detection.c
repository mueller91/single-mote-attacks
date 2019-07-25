#include "anomaly-detection.h"

#include "feature-extraction.h"
#include "helper_funcs.h"
#include "spline-detector.h"
#include "spline-data.h"

#include "sys/ctimer.h"

#include "sys/rtimer.h"

#include "net/ip/uip.h"
#include "net/ipv6/uip-ds6.h"
#include "net/linkaddr.h"

#ifdef SEND_ANP_ALERT
#include "simple-udp.h"
#endif

#include <stdio.h>

// query timer to regularily poll features and kick of Anom-Detection
static struct ctimer poll_features_timer;

// timer interval
#define QUERY_TIMER_INTERVAL 5*CLOCK_SECOND

// Send broadcast parameters
#define ANOMALY_ALERT_UDP_PORT 3001

#define ANOMALY_THRESHOLD -1500
#define ALERT_PAYLOAD_LEN 60

#ifdef SEND_ANP_ALERT
static struct simple_udp_connection broadcast_connection;

static struct uip_udp_conn *anomaly_forward_udp_conn;
static uip_ipaddr_t anomaly_srvipaddr;


static void
receiver(struct simple_udp_connection *c,
         const uip_ipaddr_t *sender_addr,
         uint16_t sender_port,
         const uip_ipaddr_t *receiver_addr,
         uint16_t receiver_port,
         const uint8_t *data,
         uint16_t datalen)
{
   static int seq_id;
    
   int alert_broadcast = check_if_contains(data, datalen, 'A', 10);
   if(alert_broadcast != 0)
   {
     char buf[ALERT_PAYLOAD_LEN];
     printf("Broadcast on port %d from port %d with length %d\n", receiver_port, sender_port, datalen);
     
     sprintf(buf, "ANOMALY ALERT %d from %s!", ++seq_id, &anomaly_forward_udp_conn->ripaddr);
     printf(" (msg: %s)\n", buf);
     uip_udp_packet_sendto(anomaly_forward_udp_conn, buf, strlen(buf), &anomaly_srvipaddr, UIP_HTONS(3002));    
   }
}


void send_anomaly_alert(float score)
{
  
  uip_ipaddr_t addr;
  uip_create_linklocal_allnodes_mcast(&addr);

  unsigned node_int_adress = lladdr_to_int(&linkaddr_node_addr, 8);

  char buf[40];
  sprintf(buf, "AAAAAAAAAAAAAAAAAAAAALERT from %d: %d ", node_int_adress, (int)score);

  simple_udp_sendto(&broadcast_connection, buf, 4, &addr);
}

#endif

/*---------------------------------------------------------------------------*/
// callback function to poll all features
void poll_all_features(void) 
{
    rtimer_clock_t start, end;
    start = RTIMER_NOW();
    //printf("start %lu\n", (unsigned long)start);
    // reset timer
    ctimer_reset(&poll_features_timer);

    // declare variables
    unsigned dis_cnt, dio_cnt, dao_cnt, vers_numbs;
    float t_f;

    // get features
    dis_cnt = get_feature_counts(DIS_FEATURE);
    dio_cnt = get_feature_counts(DIO_FEATURE);
    dao_cnt = get_feature_counts(DAO_FEATURE);
    vers_numbs = get_dodag_version_difference();
    t_f = get_feature_tf();

    unsigned num_neighbors = get_num_neighbors();

    // print
    printf("[CT]DIS: %d, DIO: %d, DAO: %d, Vers-Nums: %d, T_F: ", dis_cnt, dio_cnt, dao_cnt, vers_numbs);
    putFloat(t_f, 1000);
    printf(", NEIGHBORS: %d\n", num_neighbors);


    // Boundary case handling
    num_neighbors = num_neighbors > MAX_NEIGHBORS ? MAX_NEIGHBORS : num_neighbors;
    num_neighbors = num_neighbors == 0 ? 1 : num_neighbors;

    const struct spline_data* spline_set_ptr = select_spline_set(num_neighbors);

    float dis_score, dio_score, dao_score, vers_numbs_score, tf_score, total_score;
    dis_score = dio_score = dao_score = vers_numbs_score = tf_score = total_score = 0.0;

    dis_score = score_sample(&spline_set_ptr[DIS_FEATURE], (float) dis_cnt);
    printf("[SC]DIS Score: "); putFloat(dis_score, 1000);        // TODO do not increase precision above 100 !!
    
    dio_score = score_sample(&spline_set_ptr[DIO_FEATURE], (float) dio_cnt);
    printf(", DIO Score: "); putFloat(dio_score, 1000);       // TODO do not increase precision above 100 !!
    
    dao_score = score_sample(&spline_set_ptr[DAO_FEATURE], (float) dao_cnt);
    printf(", DAO Score: "); putFloat(dao_score, 1000);       // TODO do not increase precision above 100 !!
  
    vers_numbs_score = score_sample(&spline_set_ptr[DODAG_VERSION_FEATURE], (float) vers_numbs);
    printf(", DODAG Score: "); putFloat(vers_numbs_score, 1000);       // TODO do not increase precision above 100 !!
    
    tf_score = score_sample(&spline_set_ptr[UDP_TF_FEATURE], t_f);
    printf(", UDP_T/F Score: "); putFloat(tf_score, 1000);       // TODO do not increase precision above 100 !!
    
    total_score = dis_score + dio_score + dao_score + vers_numbs_score + tf_score;
    
    printf(", ANOMALY SCORE: "); putFloat(total_score, 1000);       // TODO do not increase precision above 100 !!
    printf(", NEIGHBORS: %d\n", num_neighbors);

#ifdef SEND_ANP_ALERT
    if(total_score < ANOMALY_THRESHOLD)
      send_anomaly_alert(total_score);
#endif
    end = RTIMER_NOW();
    unsigned long diff = 0;
    if (start > end){
        diff = (1<<16)-1 - start + end;
    }
    else {
        diff = end-start;
    }
    printf("[Q] poll_all_features: %lu\n", diff);
    //printf("start %lu\n", (unsigned long)start);
    //printf("end %lu\n", (unsigned long)end);
    //float diff_float = ((float) end)-((float) start);
    //putFloat(diff_float, 10);
    //printf("\n");
    //printf("[Q] poll_all_features: %lu\n", (unsigned long)end-start);
}

/*---------------------------------------------------------------------------*/
// the main loop for anomaly detection
void anomaly_detection_main_loop(void) 
{
#ifdef SEND_ANP_ALERT
   // Anomaly Broadcast
   simple_udp_register(&broadcast_connection, ANOMALY_ALERT_UDP_PORT, NULL, ANOMALY_ALERT_UDP_PORT, receiver);

   // Anomaly forward to root
   uip_ip6addr(&anomaly_srvipaddr,0xaaaa,0,0,0,0xc30c,0,0,0);
   anomaly_forward_udp_conn = udp_new(NULL, UIP_HTONS(3000), NULL);

   udp_bind(anomaly_forward_udp_conn, UIP_HTONS(3001)); 
#endif

   /* ---------------------------- */
   // init feature observers
   init_feature_loop(DIS_FEATURE);
   init_feature_loop(DIO_FEATURE);
   init_feature_loop(DAO_FEATURE);
   init_dodag_version_loop();
   init_feature_t_f_count_loop();

   init_neighbor_count_loop();

   // init query timer:
   ctimer_set(&poll_features_timer, QUERY_TIMER_INTERVAL, poll_all_features, NULL);

}

