#ifndef SPLINE_DETECTOR_H_
#define SPLINE_DETECTOR_H_

#define NUM_BD_POINTS 5
#define SPLINE_DEGREE 3

struct spline_data {
  float spline_coefficients[NUM_BD_POINTS-1][SPLINE_DEGREE + 1];
  float bandwidth;
  float right_boundary;
};

/**
 * @brief Evaluate the spline S at point x. Out of domain x values are extrapolated using with -x
 * @param spline: pointer to a spline struct containing the coefficients and interval boundary points
 * @param x: point at which the spline
 * @return S(x) for x inside the domain of x, S(x_n) - x otherwise with x_n being the right domain boundary
 */
float score_sample(const struct spline_data* spline, float x);

#endif /* SPLINE_DETECTOR_H_ */