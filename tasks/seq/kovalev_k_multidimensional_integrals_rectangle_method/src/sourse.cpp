#include "seq/kovalev_k_multidimensional_integrals_rectangle_method/include/header.hpp"

bool kovalev_k_multidimensional_integrals_rectangle_method_seq::MultidimensionalIntegralsRectangleMethod::
    count_multidimensional_integrals_rectangle_method_seq() {
  std::stack<std::vector<double>> stack;
  stack.emplace();

  while (!stack.empty()) {
    std::vector<double> point = stack.top();
    stack.pop();

    if (point.size() == n) {
      res += func(point) * std::pow(h, n);
      continue;
    }

    int dim = point.size();

    for (double x = limits[dim].first; x + h <= limits[dim].second; x += h) {
      point.push_back(x + h / 2);
      stack.emplace(point);
      point.pop_back();
    }
  }

  return true;
}

bool kovalev_k_multidimensional_integrals_rectangle_method_seq::MultidimensionalIntegralsRectangleMethod::
    pre_processing() {
  internal_order_test();
  fflush(stdout);
  limits.resize(n);
  void* ptr_input = taskData->inputs[0];
  void* ptr_vec = limits.data();
  h = reinterpret_cast<double*>(taskData->inputs[1])[0];
  memcpy(ptr_vec, ptr_input, sizeof(std::pair<double, double>) * n);
  res = 0;
  return true;
}

bool kovalev_k_multidimensional_integrals_rectangle_method_seq::MultidimensionalIntegralsRectangleMethod::validation() {
  internal_order_test();
  return (taskData->inputs_count[0] > 0 && taskData->outputs_count[0] == 1 && taskData->inputs_count[0] == n &&
          reinterpret_cast<double*>(taskData->inputs[1])[0] <= 0.01);
}

bool kovalev_k_multidimensional_integrals_rectangle_method_seq::MultidimensionalIntegralsRectangleMethod::run() {
  internal_order_test();
  return count_multidimensional_integrals_rectangle_method_seq();
}

bool kovalev_k_multidimensional_integrals_rectangle_method_seq::MultidimensionalIntegralsRectangleMethod::
    post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = res;
  return true;
}