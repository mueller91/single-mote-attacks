#ifndef PACKAGE_PARSER_H_
#define PACKAGE_PARSER_H_

#include "core/net/linkaddr.h"


void parse_adress(const linkaddr_t *fromAddr, const linkaddr_t *toAddr);


void parse_package(const linkaddr_t *fromAddr,
                             const linkaddr_t *toAddr,
                             const uint16_t size,
                             const uint8_t *hdrptr);


#endif /* PACKAGE_PARSER_H_ */