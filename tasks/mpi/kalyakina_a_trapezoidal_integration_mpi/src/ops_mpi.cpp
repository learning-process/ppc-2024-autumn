// Copyright 2023 Nesterov Alexander
#include "mpi/kalyakina_a_trapezoidal_integration_mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <thread>
#include <vector>

unsigned int kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskSequential::CalculationOfCoefficient(
    const std::vector<double>& point) {
  unsigned int degree = limits.size();
  for (unsigned int i = 0; i < limits.size(); i++) {
    if ((limits[i].first == point[i]) || (limits[i].second == point[i])) {
      degree--;
    }
  }

  return pow(2, degree);
}

void kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskSequential::Recursive(
    std::vector<double>& _point, unsigned int& definition, unsigned int divider, unsigned int variable) {
  if (variable > 0) {
    Recursive(_point, definition, divider * (number_of_intervals[variable] + 1), variable - 1);
  }
  _point[variable] = limits[variable].first + definition / divider *
                                                  (limits[variable].second - limits[variable].first) /
                                                  number_of_intervals[variable];
  definition = definition % divider;
}

std::vector<double> kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskSequential::GetPointFromNumber(
    unsigned int number) {
  std::vector<double> point(limits.size());
  unsigned int definition = number;
  Recursive(point, definition, 1, limits.size() - 1);

  return point;
}

bool kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskSequential::pre_processing() {
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

bool kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskSequential::validation() {
  internal_order_test();

  return ((taskData->inputs.size() == 3) && (taskData->inputs_count[0] == 1) &&
          (reinterpret_cast<unsigned int*>(taskData->inputs[0])[0] == taskData->inputs_count[1]) &&
          (taskData->inputs_count[1] == taskData->inputs_count[2]) && (taskData->outputs_count[0] == 1));
}

bool kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskSequential::run() {
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

bool kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = result;

  return true;
}

unsigned int kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel::CalculationOfCoefficient(
    const std::vector<double>& point) {
  unsigned int degree = limits.size();
  for (unsigned int i = 0; i < limits.size(); i++) {
    if ((limits[i].first == point[i]) || (limits[i].second == point[i])) {
      degree--;
    }
  }

  return pow(2, degree);
}

void kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel::Recursive(std::vector<double>& _point,
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

std::vector<double> kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel::GetPointFromNumber(
    unsigned int number) {
  std::vector<double> point(limits.size());
  unsigned int definition = number;
  Recursive(point, definition, 1, limits.size() - 1);

  return point;
}

bool kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    limits = std::vector<std::pair<double, double>>(taskData->inputs_count[1]);
    auto* it1 = reinterpret_cast<std::pair<double, double>*>(taskData->inputs[1]);
    std::copy(it1, it1 + taskData->inputs_count[1], limits.begin());

    number_of_intervals = std::vector<unsigned int>(taskData->inputs_count[2]);
    auto* it2 = reinterpret_cast<unsigned int*>(taskData->inputs[2]);
    std::copy(it2, it2 + taskData->inputs_count[2], number_of_intervals.begin());

    result = 0.0;
  }

  return true;
}

bool kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel::validation() {
  internal_order_test();

  return (world.rank() != 0) ||
         ((taskData->inputs.size() == 3) && (taskData->inputs_count[0] == 1) &&
          (reinterpret_cast<unsigned int*>(taskData->inputs[0])[0] == taskData->inputs_count[1]) &&
          (taskData->inputs_count[1] == taskData->inputs_count[2]) && (taskData->outputs_count[0] == 1));
}

bool kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel::run() {
  internal_order_test();

  boost::mpi::broadcast(world, limits, 0);
  boost::mpi::broadcast(world, number_of_intervals, 0);

  std::vector<unsigned int> count_of_points;
  std::vector<unsigned int> first_point_numbers;
  unsigned int local_count_of_points;
  unsigned int local_first_point_numbers;
  double local_result = 0.0;

  if (world.rank() == 0) {
    count_of_points.resize(world.size());
    first_point_numbers.resize(world.size());
    unsigned int delta = 1;
    unsigned int current_number = 0;
    for (unsigned int i = 0; i < number_of_intervals.size(); i++) {
      delta *= (number_of_intervals[i] + 1);
    }
    unsigned int remainder = delta % world.size();
    delta /= world.size();
    for (unsigned int i = 0; i < world.size() - remainder; i++) {
      count_of_points[i] = delta;
      first_point_numbers[i] = current_number;
      current_number += delta;
    }
    delta++;
    for (int i = world.size() - remainder; i < world.size(); i++) {
      count_of_points[i] = delta;
      first_point_numbers[i] = current_number;
      current_number += delta;
    }
  }

  boost::mpi::scatter(world, count_of_points.data(), &local_count_of_points, 1, 0);
  boost::mpi::scatter(world, first_point_numbers.data(), &local_first_point_numbers, 1, 0);

  for (unsigned int i = 0; i < local_count_of_points; i++) {
    std::vector<double> point = GetPointFromNumber(local_first_point_numbers + i);
    local_result += CalculationOfCoefficient(point) * function(point);
  }

  boost::mpi::reduce(world, local_result, result, std::plus(), 0);

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < limits.size(); i++) {
      result *= (limits[i].second - limits[i].first) / number_of_intervals[i];
    }
    result /= pow(2, limits.size());
  }

  return true;
}

bool kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result;
  }

  world.barrier();
  return true;
}
