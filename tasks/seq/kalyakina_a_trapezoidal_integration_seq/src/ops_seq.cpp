// Copyright 2023 Nesterov Alexander
#include "seq/kalyakina_a_trapezoidal_integration_seq/include/ops_seq.hpp"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

unsigned int kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask::CalculationOfCoefficient(
    const std::vector<double>& point) {
  unsigned int degree = limits.size();
  for (unsigned int i = 0; i < limits.size(); i++) {
    if ((limits[i].first == point[i]) || (limits[i].second == point[i])) {
      degree--;
    }
  }

  return pow(2, degree);
}

void kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask::Recursive(std::vector<double>& _point,
                                                                                    unsigned int& definition,
                                                                                    unsigned int divider,
                                                                                    unsigned int variable) {
  if (variable > 0) {
    Recursive(_point, definition, divider * (number_of_intervals[variable] + 1), variable - 1);
  }
  _point[variable] = limits[variable].first + definition / divider *
                                                  (limits[variable].second - limits[variable].first) /
                                                  number_of_intervals[variable];
  definition = definition % divider;
}

std::vector<double> kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask::GetPointFromNumber(
    unsigned int number) {
  std::vector<double> point(limits.size());
  unsigned int definition = number;
  Recursive(point, definition, 1, limits.size() - 1);

  return point;
}

bool kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask::pre_processing() {
  internal_order_test();

  limits = std::vector<std::pair<double, double>>(taskData->inputs_count[1]);
  auto* it1 = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[1]);
  std::copy(it1, it1 + taskData->inputs_count[1], limits.begin());

  number_of_intervals = std::vector<unsigned int>(taskData->inputs_count[2]);
  auto* it2 = reinterpret_cast<unsigned int*>(taskData->inputs[2]);
  std::copy(it2, it2 + taskData->inputs_count[2], number_of_intervals.begin());

  result = 0.0;

  return true;
}

bool kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask::validation() {
  internal_order_test();

  return ((taskData->inputs.size() == 3) && (taskData->inputs_count[0] == 1) &&
          (reinterpret_cast<unsigned int*>(taskData->inputs[0])[0] == taskData->inputs_count[1]) &&
          (taskData->inputs_count[1] == taskData->inputs_count[2]) && (taskData->outputs_count[0] == 1));
}

bool kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask::run() {
  internal_order_test();

  unsigned int count = 1;
  for (unsigned int i = 0; i < number_of_intervals.size(); i++) {
    count *= (number_of_intervals[i] + 1);
  }

  for (unsigned int i = 0; i < count; i++) {
    std::vector<double> point = GetPointFromNumber(i);
    result += CalculationOfCoefficient(point) * function(point);
  }

  for (unsigned int i = 0; i < limits.size(); i++) {
    result *= (limits[i].second - limits[i].first) / number_of_intervals[i];
  }

  result /= pow(2, limits.size());

  return true;
}

bool kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;

  return true;
}
