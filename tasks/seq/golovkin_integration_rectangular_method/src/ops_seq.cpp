// Golovkin Maksim

#include "seq/golovkin_integration_rectangular_method/include/ops_seq.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace golovkin_integration_rectangular_method;

IntegralCalculator::IntegralCalculator(const std::shared_ptr<ppc::core::TaskData>& taskData)
    : ppc::core::Task(taskData),
      taskData(taskData),
      a(0.0),
      b(0.0),
      epsilon(0.01),
      cnt_of_splits(0),
      h(0.0),
      res(0.0) {}

bool IntegralCalculator::validation() {
  internal_order_test();
  return (taskData->inputs_count.size() == 2 && taskData->inputs_count[0] >= 0 && taskData->inputs_count[1] >= 0 &&
          taskData->outputs_count[0] == taskData->inputs_count[0]);
}

bool IntegralCalculator::pre_processing() {
  internal_order_test();
  try {
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
  } catch (const std::exception& e) {
    std::cerr << "Error converting input data: " << e.what() << std::endl;
    return false;
  }

  if (a == b) {
    res = 0.0;
    return true;
  }

  cnt_of_splits = static_cast<int>(std::abs(b - a) / epsilon);

  h = (b - a) / cnt_of_splits;
  return true;
}

bool IntegralCalculator::run() {
  internal_order_test();
  double result = 0.0;

  if (a == b) {
    return true;
  }

  for (int i = 0; i < cnt_of_splits; ++i) {
    double x = a + (i + 0.5) * h;
    result += function_square(x);
  }
  result *= h;
  res = result;
  return true;
}

bool IntegralCalculator::post_processing() {
  internal_order_test();
  try {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  } catch (const std::exception& e) {
    std::cerr << "Error writing output data: " << e.what() << std::endl;
  }
  return true;
}

double IntegralCalculator::function_square(double x) { return x * x; }