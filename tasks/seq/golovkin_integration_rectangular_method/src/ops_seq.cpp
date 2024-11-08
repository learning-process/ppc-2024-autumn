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
  if (taskData->inputs.size() != 3) {
    std::cerr << "Error: 3 inputs were expected, but received " << taskData->inputs.size() << std::endl;
  }
  if (taskData->outputs.size() != 1) {
    std::cerr << "Error: 1 output was expected, but received " << taskData->outputs.size() << std::endl;
  }
  return true;
}

bool IntegralCalculator::pre_processing() {
  try {
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    epsilon = *reinterpret_cast<double*>(taskData->inputs[2]);
  } catch (const std::exception& e) {
    std::cerr << "Error converting input data: " << e.what() << std::endl;
    return false;
  }

  cnt_of_splits = static_cast<int>(std::abs(b - a) / epsilon);

  h = (b - a) / cnt_of_splits;
  return true;
}

bool IntegralCalculator::run() {
  double result = 0.0;

  for (int i = 0; i < cnt_of_splits; ++i) {
    double x = a + (i + 0.5) * h;
    result += function_square(x);
  }
  result *= h;
  res = result;
  return true;
}

bool IntegralCalculator::post_processing() {
  try {
    *reinterpret_cast<double*>(taskData->outputs[0]) = res;
  } catch (const std::exception& e) {
    std::cerr << "Error writing output data: " << e.what() << std::endl;
  }
  return true;
}

double IntegralCalculator::function_square(double x) { return x * x; }