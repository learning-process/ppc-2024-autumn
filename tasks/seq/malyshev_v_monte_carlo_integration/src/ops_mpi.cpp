#include "seq/malyshev_v_monte_carlo_integration/include/ops_mpi.hpp"

#include <random>

namespace malyshev_v_monte_carlo_integration {

bool TestMPITaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 3 && taskData->outputs.size() == 1);
}

bool TestMPITaskSequential::pre_processing() {
  internal_order_test();
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  double input_epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
  epsilon = input_epsilon;
  num_samples = static_cast<int>(100 / epsilon);
  return true;
}

bool TestMPITaskSequential::run() {
  internal_order_test();
  double h = (b - a) / num_samples;
  double sum = (function_square(a) + function_square(b)) / 2.0;

  for (int i = 1; i < num_samples; ++i) {
    sum += function_square(a + i * h);
  }

  res = h * sum;
  return true;
}

bool TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}

}  // namespace malyshev_v_monte_carlo_integration