#include "seq/bessonov_e_integration_monte_carlo/include/ops_seq.hpp"

bool bessonov_e_integration_monte_carlo_seq::TestTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs.size() == 3 && taskData->outputs.size() == 1);
}

bool bessonov_e_integration_monte_carlo_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  a = *reinterpret_cast<double*>(taskData->inputs[0]);
  b = *reinterpret_cast<double*>(taskData->inputs[1]);
  num_points = *reinterpret_cast<int*>(taskData->inputs[2]);
  return true;
}

bool bessonov_e_integration_monte_carlo_seq::TestTaskSequential::run() {
  internal_order_test();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(a, b);

  double sum = 0.0;
  for (int i = 0; i < num_points; ++i) {
    double x = dis(gen);
    sum += exampl_func(x);
  }

  res = (b - a) * (sum / num_points);
  return true;
}

bool bessonov_e_integration_monte_carlo_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}
