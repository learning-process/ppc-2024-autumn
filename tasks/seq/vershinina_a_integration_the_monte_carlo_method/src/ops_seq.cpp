#include "seq/vershinina_a_integration_the_monte_carlo_method/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <vector>

bool vershinina_a_integration_the_monte_carlo_method::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_ = reinterpret_cast<double*>(taskData->inputs[0]);
  xmin = input_[0];
  xmax = input_[1];
  ymin = input_[2];
  ymax = input_[3];
  iter_count = static_cast<int>(input_[4]);
  return true;
}

bool vershinina_a_integration_the_monte_carlo_method::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 1;
}

bool vershinina_a_integration_the_monte_carlo_method::TestTaskSequential::run() {
  internal_order_test();
  int count;
  double total = 0;
  double inBox = 0;
  reference_res = 0;
  for (count = 0; count < iter_count; count++) {
    double u1 = (double)rand() / (double)RAND_MAX;
    double u2 = (double)rand() / (double)RAND_MAX;

    double xcoord = ((xmax - xmin) * u1) + xmin;
    double ycoord = ((ymax - ymin) * u2) + ymin;
    double val = p(xcoord);

    ++total;

    if (val > ycoord) {
      ++inBox;
    }
  }
  double density = inBox / total;

  reference_res = (xmax - xmin) * (ymax - ymin) * density;
  return true;
}
bool vershinina_a_integration_the_monte_carlo_method::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = reference_res;
  return true;
}
