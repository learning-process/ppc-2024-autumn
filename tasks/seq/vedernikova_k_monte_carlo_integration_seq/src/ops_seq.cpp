#include "seq/vedernikova_k_monte_carlo_integration_seq/include/ops_seq.hpp"

#include <cmath>
#include <functional>

using namespace std::chrono_literals;

bool vedernikova_k_monte_carlo_integration_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output (integration limits and result)
  a_x = *reinterpret_cast<double*>(taskData->inputs[0]);
  b_x = *reinterpret_cast<double*>(taskData->inputs[1]);
  a_y = *reinterpret_cast<double*>(taskData->inputs[2]);
  b_y = *reinterpret_cast<double*>(taskData->inputs[3]);
  num_points = *reinterpret_cast<int*>(taskData->inputs[4]);
  res = *reinterpret_cast<double*>(taskData->outputs[0]);
  return true;
}

bool vedernikova_k_monte_carlo_integration_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[0] == 5 && taskData->outputs_count[0] == 1;
}

bool vedernikova_k_monte_carlo_integration_seq::TestTaskSequential::run() {
  internal_order_test();
  double x_limit = b_x - a_x;
  double y_limit = b_y - a_y;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_x(a_x, b_x);
  std::uniform_real_distribution<> dis_y(a_y, b_y);
  double sum = 0.0;
  for (int i = 0; i < num_points; i++) {
    double x = dis_x(gen);
    double y = dis_y(gen);
    sum += f(x, y);
  }
  res = x_limit * y_limit * sum / num_points;
  return true;
}

bool vedernikova_k_monte_carlo_integration_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
