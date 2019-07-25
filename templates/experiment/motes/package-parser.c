#include "package-parser.h"

#include "helper_funcs.h"
#include "feature-extraction.h"

#include "core/net/linkaddr.h"
#include "core/net/packetbuf.h"


#define UNCOMPRESSED 0x41

#define IPHC_HEADER 0x7a
#define IPHC_ADDRESS_UDP 0x00
#define IPHC_ADDRESS_DIO 0x33 
#define IPHC_ADDRESS_DIO_MC 0x3b // Multicast compression packages
#define IPHC_ADDRESS_DAO 0x33

#define ICMP6_TYPE_RPL 0x9b
#define RPL_TYPE_DIS 0x00
#define RPL_TYPE_DIO 0x01
#define RPL_TYPE_DAO 0x02

#define UDP_OFFSET 35
#define RPL_OFFSET_COMPRESSED 3 // 4 for Multicast compression packages
#define RPL_OFFSET_UNCOMPRESSED 41

void parse_package(const linkaddr_t *fromAddr, const linkaddr_t *toAddr, const uint16_t size, const uint8_t *hdrptr)
{
    #if FEATURE_DEBUG_PACKAGEDUMP 
        printf("\nPACKAGEDUMP_BEGIN>>>>>");
        int i;
        for (i = 0; i < size; i++)
             printf("0x%02X ", hdrptr[i]);
        printf("<<<<<PACKAGEDUMP_END\n");
    #endif
    
    unsigned to_int_address = lladdr_to_int(toAddr, 8);
    unsigned from_int_address = lladdr_to_int(fromAddr, 8);
    unsigned node_int_adress = lladdr_to_int(&linkaddr_node_addr, 8);

    unsigned to_node = linkaddr_cmp(packetbuf_addr(PACKETBUF_ADDR_RECEIVER), &linkaddr_node_addr) || packetbuf_holds_broadcast();

    neighborhood_watch_handler(to_int_address, from_int_address, node_int_adress);


    int udp = check_if_contains(hdrptr, size, 'f', 10);

    if(udp > 0)
    {
        increment_udp_counts(to_int_address, from_int_address);

        #if FEATURE_DEBUG 
            printf("UDP from: ");
            net_debug_lladdr_print(fromAddr);
            printf( " to ");
            net_debug_lladdr_print(toAddr);
            printf(" with size: %d\n", size);
        #endif

        return;
    }

    if(hdrptr[0] == IPHC_HEADER)
    {
  /*      if(hdrptr[1] == IPHC_ADDRESS_UDP && size > (UDP_OFFSET + 2))
            if(hdrptr[UDP_OFFSET] == 0x11 && hdrptr[UDP_OFFSET + 1] == 0x00 && hdrptr[UDP_OFFSET + 2] == 0x63)
            {
                increment_udp_counts(to_int_address, from_int_address);
                return;
            }*/
        
        if(to_node && hdrptr[1] == IPHC_ADDRESS_DAO && size > (RPL_OFFSET_COMPRESSED + 1))
            if(hdrptr[RPL_OFFSET_COMPRESSED] == ICMP6_TYPE_RPL && hdrptr[RPL_OFFSET_COMPRESSED + 1] == RPL_TYPE_DAO)
            {
                detector_packagecount_handler(DAO_FEATURE);
                return;
            }
        
        if(to_node && (hdrptr[1] == IPHC_ADDRESS_DIO || hdrptr[1] == IPHC_ADDRESS_DIO_MC) && size > (RPL_OFFSET_COMPRESSED + 6))
        {
            uint8_t rpl_offset = hdrptr[1] == IPHC_ADDRESS_DIO_MC ? (RPL_OFFSET_COMPRESSED + 1) : RPL_OFFSET_COMPRESSED;
            
            if(hdrptr[rpl_offset] == ICMP6_TYPE_RPL && hdrptr[rpl_offset + 1] == RPL_TYPE_DIO)
            {
                detector_packagecount_handler(DIO_FEATURE);
                
                uint8_t rpl_version = hdrptr[rpl_offset + 5];
                uint8_t rpl_rank = hdrptr[rpl_offset + 6];
                
                call_detector_dodag_version_handler(rpl_version);
                return;
            }    
        }
    }
    
    if(to_node && hdrptr[0] == UNCOMPRESSED)
        if(hdrptr[1] == 0x60 && size > (RPL_OFFSET_UNCOMPRESSED + 2))
            if(hdrptr[RPL_OFFSET_UNCOMPRESSED] == ICMP6_TYPE_RPL && hdrptr[RPL_OFFSET_UNCOMPRESSED + 1] == RPL_TYPE_DIS)
            {
                detector_packagecount_handler(DIS_FEATURE);
                return;
            }
    
    return;
}