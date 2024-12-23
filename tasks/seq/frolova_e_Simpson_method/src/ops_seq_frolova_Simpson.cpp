// Copyright 2024 Nesterov Alexander
#include "seq/frolova_e_Simpson_method/include/ops_seq_frolova_Simpson.hpp"

#include <thread>

using namespace std::chrono_literals;

double frolova_e_Simpson_method_seq::roundToTwoDecimalPlaces(double value) { return std::round(value * 100.0) / 100.0; }

double frolova_e_Simpson_method_seq::squaresOfX(const std::vector<double>& point) {
  double x = point[0];
  return x * x;
}

double frolova_e_Simpson_method_seq::cubeOfX(const std::vector<double>& point) {
  double x = point[0];
  return x * x * x;
}

double frolova_e_Simpson_method_seq::sumOfSquaresOfXandY(const std::vector<double>& point) {
  double x = point[0];
  double y = point[1];
  return x * x + y * y;
}

double frolova_e_Simpson_method_seq::ProductOfXAndY(const std::vector<double>& point) {
  double x = point[0];
  double y = point[1];
  return x * y;
}

double frolova_e_Simpson_method_seq::sumOfSquaresOfXandYandZ(const std::vector<double>& point) {
  double x = point[0];
  double y = point[1];
  double z = point[2];
  return x * x + y * y + z * z;
}

double frolova_e_Simpson_method_seq::ProductOfSquaresOfXandYandZ(const std::vector<double>& point) {
  double x = point[0];
  double y = point[1];
  double z = point[2];
  return x * y * z;
}

double frolova_e_Simpson_method_seq::Simpson_Method(double (*func)(const std::vector<double>&), size_t divisions,
                                                    size_t dimension, std::vector<double>& limits) {
  std::vector<double> h(dimension);
  std::vector<int> steps(dimension);
  std::vector<int> nodes(dimension);
  std::vector<int> offset(dimension);

  std::vector<double> grid;
  int totalPoints = 0;

  for (size_t i = 0; i < dimension; ++i) {
    double a = limits[2 * i];
    double b = limits[2 * i + 1];

    steps[i] = divisions;
    nodes[i] = steps[i] + 1;
    h[i] = (b - a) / steps[i];

    offset[i] = totalPoints;

    for (int j = 0; j < nodes[i]; ++j) {
      grid.push_back(a + j * h[i]);
    }

    totalPoints += nodes[i];
  }

  std::vector<int> indices(dimension, 0);
  std::vector<double> point(dimension);
  double integral = 0.0;

  int totalIterations = 1;
  for (size_t i = 0; i < dimension; ++i) {
    totalIterations *= nodes[i];
  }

  for (int linearIndex = 0; linearIndex < totalIterations; ++linearIndex) {
    int temp = linearIndex;

    for (size_t i = 0; i < dimension; ++i) {
      indices[i] = temp % nodes[i];
      temp /= nodes[i];
    }

    for (size_t i = 0; i < dimension; ++i) {
      point[i] = grid[offset[i] + indices[i]];
    }

    double weight = 1.0;
    for (size_t i = 0; i < dimension; ++i) {
      if (indices[i] == 0 || indices[i] == steps[i])
        weight *= 1.0;
      else if (indices[i] % 2 == 1)
        weight *= 4.0;
      else
        weight *= 2.0;
    }

    integral += weight * func(point);
  }

  for (size_t i = 0; i < dimension; ++i) {
    integral *= h[i] / 3.0;
  }

  return roundToTwoDecimalPlaces(integral);
}

bool frolova_e_Simpson_method_seq::Simpsonmethod::pre_processing() {
  internal_order_test();

  int* value = reinterpret_cast<int*>(taskData->inputs[0]);
  divisions = static_cast<size_t>(value[0]);
  dimension = static_cast<size_t>(value[1]);

  double* value_2 = reinterpret_cast<double*>(taskData->inputs[1]);
  for (int i = 0; i < static_cast<int>(taskData->inputs_count[1]); i++) {
    limits.push_back(value_2[i]);
  }

  return true;
}

bool frolova_e_Simpson_method_seq::Simpsonmethod::validation() {
  internal_order_test();

  int* value = reinterpret_cast<int*>(taskData->inputs[0]);
  if (taskData->inputs_count[0] != 2) {
    return false;
  }

  auto div = static_cast<size_t>(value[0]);
  if (static_cast<int>(div) % 2 != 0) {
    return false;
  }

  auto dim = static_cast<size_t>(value[1]);
  if (taskData->inputs_count[1] / dim != 2) {
    return false;
  } 
  return true;
}

bool frolova_e_Simpson_method_seq::Simpsonmethod::run() {
  internal_order_test();
  resIntegral = Simpson_Method(func, divisions, dimension, limits);

  return true;
}

bool frolova_e_Simpson_method_seq::Simpsonmethod::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = resIntegral;

  return true;
}