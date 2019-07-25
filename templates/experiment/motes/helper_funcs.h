#ifndef HELPER_FUNCS_H
#define HELPER_FUNCS_H

#include "core/net/linkaddr.h"
#include "core/net/ip/uip.h"

#if FEATURE_DEBUG
	/*---------------------------------------------------------------------------*/
	void multidim_col_sum(int n_rows, int n_cols, int arr[n_rows][n_cols], int *result);

	/*---------------------------------------------------------------------------*/
	void print_array(int *a, int size);

	/*---------------------------------------------------------------------------*/
	void print_multi_array(int n_rows, int n_cols, int arr[n_rows][n_cols]);

#endif

/*---------------------------------------------------------------------------*/
int check_if_contains( const uint8_t *hdrptr, const uint16_t size, const char c, const int times);

// Workaround for printing floats
void putLong(long x);
void putFloat(float f, unsigned p);
	
/*---------------------------------------------------------------------------*/
int lladdr_to_int(const uip_lladdr_t *addr, const int SIZE);

/*---------------------------------------------------------------------------*/
#endif /*  HELPER_FUNCS_H */