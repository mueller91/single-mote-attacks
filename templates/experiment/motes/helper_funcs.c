#include "helper_funcs.h"
#include "core/net/ip/uip.h"

#include <stdio.h>
#include <limits.h>

/*---------------------------------------------------------------------------*/
int lladdr_to_int(const uip_lladdr_t *addr, const int SIZE ) {
    if (SIZE == 8) {

    unsigned int i;
    unsigned int res = 0;

    for(i = 0; i < SIZE; i++) {
      res = 100 * res + addr->addr[i];
    }
    return res;
  }
  else
  {
    printf("NOT IMPLEMENTED: lladdr_to_int!");
    return -1;
  }
}


void putLong(long x)
{
    if(x < 0)
    {
        putchar('-');
        x = -x;
    }
    if (x >= 10) 
    {
        putLong(x / 10);
    }
    putchar(abs(x) % 10+'0');
}


void putFloat(float f, unsigned p)
{
    // if f > than max value of long
    if(fabs(f) >= LONG_MAX)
    {
      if(f < 0)
         putchar('-');
     
      printf("%ld", LONG_MAX);      
    }
    else
    {
      long i = (long)f;
      putLong(i);

      if (p > 0) {
          f = (f - i) * p;
          i = abs((long)f);

          // if we want to print decimals
          if( fabs(f) - i >= 0.5f )
          {
              i++ ;
          }

          putchar('.') ;

          // TODO if p == 10000, produces outputs such as -815.-32767
          putLong( i ) ;
      }
    }
}


  /*---------------------------------------------------------------------------*/
  /* Check if char c is contained at least times times in [hdrptr, hdrptr + size]*/
  int check_if_contains( const uint8_t *hdrptr, const uint16_t size, const char c, const int times) {

          int i;
          int consecutives = 0;
          for (i = 0; i < size; i++)
          {
             if ((const unsigned char) c == ((const unsigned char *) hdrptr)[i]) {
                  consecutives++;
  //                printf("%c == %c\n", (const unsigned char *) hdrptr[i], c);
             } else {
                  consecutives = 0;
  //                printf("%c =!= %c\n", (const unsigned char *) hdrptr[i], c);
             }
             if (consecutives > times - 1) {
                  return 1;
             }
          }
          return 0;
  }

#if FEATURE_DEBUG
  /*---------------------------------------------------------------------------*/
  // helper function to caluclate the sum of columns
  void multidim_col_sum(int n_rows, int n_cols, int arr[n_rows][n_cols], int *result) {
      int row, col;
      for (row = 0; row < n_rows; row += 1) {
         int colsum = 0;
         for (col = 0; col < n_cols; col += 1)
          {
               colsum += arr[row][col];
          }
          result[row] = colsum;
      }
  }

  // https://stackoverflow.com/questions/9280654/c-printing-bits
  void print_bit_array(unsigned vec)
  {

    // get constants
    const unsigned size = sizeof(unsigned);
    unsigned selector = 1;
    unsigned num = 0;

    printf(" Bits Set: [");
    int i=0,j;
    // for each byte
    for(;i<size;++i){
        // for each bit per byte
        for(j=0;j<8;++j){
            // print last bit and shift left.
            if (vec & selector) {
               printf("%u, ",i*8+j);
               num = num + 1;
               } else {
               printf("-, ");
               }
            selector = selector << 1;
            }
        }
        printf("] (%d)\n", num);
    }

  /*---------------------------------------------------------------------------*/
  void print_array(int *a, int size) {
      int i;
      for (i = 0; i < size; i += 1) {
          printf("%d, ", a[i]);
      }
      printf("\n");
  }

  /*---------------------------------------------------------------------------*/
  void print_multi_array(int n_rows, int n_cols, int arr[n_rows][n_cols]) {
      int i, j;
      for (j = 0; j < n_rows; j += 1) {
          for (i = 0; i < n_cols; i += 1) {
              printf("%d, ", arr[j][i]);
          }
          printf("\n");
      }
  }


#endif /* FEATURE_DEBUG */


















