/*---------------------------------------------------------------------------*/
/*  PROJECT CONF FILE */
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
#ifndef PROJECT_CONF_H_
#define PROJECT_CONF_H_
/*---------------------------------------------------------------------------*/
#undef NETSTACK_CONF_RDC
#define NETSTACK_CONF_RDC contikimac_detector_driver

//#undef NETSTACK_CONF_NETWORK
//#define NETSTACK_CONF_NETWORK uip_driver
//
#define DEBUG 1
//
//#define NETSTACK_CONF_MAC nullmac_driver
//#define NETSTACK_CONF_RDC contikimac_detector_driver
//#define NETSTACK_CONF_FRAMER framer_nullmac

/*---------------------------------------------------------------------------*/
#endif /* PROJECT_CONF_H_ */