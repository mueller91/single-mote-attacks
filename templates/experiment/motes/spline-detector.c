
#include "spline-detector.h"

typedef enum { false, true } bool;


float score_sample(const struct spline_data* spline, float x)
{
  bool extrapolate = true;
  unsigned spline_number = NUM_BD_POINTS - 1;
  
  float h = spline->right_boundary / (NUM_BD_POINTS - 1);

  // Find correct subinterval
  unsigned i;
  for(i=1; i < NUM_BD_POINTS; ++i)
  {
    if( x < i * h)
    {
      spline_number = i-1;
      extrapolate = false;
      break;
    } 
  }
  
  // Evaluate the the spline on the correct subinterval
  const float* coefficients = extrapolate == false ? spline->spline_coefficients[spline_number] : spline->spline_coefficients[spline_number - 1];

  // Interpolate: Evaluate relative to distance to subinterval starting point
  // Extrapolate: Calculate value at the last boundary point (distance of h away from second to last)
  float x_translated = extrapolate == false ? x - spline_number * h : h;
  
  float result = 0;
  float x_powers = 1;
     
  for(i = 0; i <= SPLINE_DEGREE; ++i)
  {
        float coeff = coefficients[SPLINE_DEGREE-i];
        float monomial_result = coeff * x_powers;
        result += monomial_result;

        x_powers *= x_translated; 
  }
  
  // Assume gaussian decay of corresponding density
  if (extrapolate == true)
  {
      float bandwidth = spline->bandwidth;
      x_translated = x - spline_number * h; // Distance to the end of the full interpolation domain
      float x_trans_boundary = 64.;
      x_translated = (x_translated >= x_trans_boundary) ? x_trans_boundary : x_translated;
      float bandwidth_boundary = 1./32.;
      bandwidth = (bandwidth <= bandwidth_boundary) ? bandwidth_boundary : bandwidth;
      result -= 0.5 * x_translated * x_translated / (bandwidth * bandwidth);
  }
    
  return result;
}
