// Copyright 2023 Nesterov Alexander
#include "mpi/frolova_e_Simpson_method/include/ops_mpi_frolova_Simpson.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

double frolova_e_Simpson_method_mpi::squaresOfX(const std::vector<double>& point) {
  double x = point[0];
  return x * x;
}

double frolova_e_Simpson_method_mpi::cubeOfX(const std::vector<double>& point) {
  double x = point[0];
  return x * x * x;
}

double frolova_e_Simpson_method_mpi::sumOfSquaresOfXandY(const std::vector<double>& point) {
  double x = point[0];
  double y = point[1];
  return x * x + y * y;
}

double frolova_e_Simpson_method_mpi::ProductOfXAndY(const std::vector<double>& point) {
  double x = point[0];
  double y = point[1];
  return x * y;
}

double frolova_e_Simpson_method_mpi::sumOfSquaresOfXandYandZ(const std::vector<double>& point) {
  double x = point[0];
  double y = point[1];
  double z = point[2];
  return x * x + y * y + z * z;
}

double frolova_e_Simpson_method_mpi::ProductOfSquaresOfXandYandZ(const std::vector<double>& point) {
  double x = point[0];
  double y = point[1];
  double z = point[2];
  return x * y * z;
}

double frolova_e_Simpson_method_mpi::roundToTwoDecimalPlaces(double value) { return std::round(value * 100.0) / 100.0; }

double frolova_e_Simpson_method_mpi::Simpson_Method(double (*func)(const std::vector<double>&), size_t divisions,
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

bool frolova_e_Simpson_method_mpi::SimpsonmethodSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  int* value = reinterpret_cast<int*>(taskData->inputs[0]);  //{divisions,dimension}
  divisions = static_cast<size_t>(value[0]);
  dimension = static_cast<size_t>(value[1]);

  auto* value_2 = reinterpret_cast<double*>(taskData->inputs[1]);
  for (int i = 0; i < static_cast<int>(taskData->inputs_count[1]); i++) {
    limits.push_back(value_2[i]);
  }
  return true;
}

bool frolova_e_Simpson_method_mpi::SimpsonmethodSequential::validation() {
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

  return taskData->inputs_count[1] / dim == 2;
}

bool frolova_e_Simpson_method_mpi::SimpsonmethodSequential::run() {
  internal_order_test();
  resIntegral = Simpson_Method(func, divisions, dimension, limits);

  return true;
}

bool frolova_e_Simpson_method_mpi::SimpsonmethodSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = resIntegral;

  return true;
}

bool frolova_e_Simpson_method_mpi::SimpsonmethodParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    // Init value for input and output
    int* value = reinterpret_cast<int*>(taskData->inputs[0]);  //{divisions,dimension,functionid}
    divisions = static_cast<size_t>(value[0]);
    dimension = static_cast<size_t>(value[1]);
    functionid = static_cast<size_t>(value[2]);

    auto* value_2 = reinterpret_cast<double*>(taskData->inputs[1]);
    for (int i = 0; i < static_cast<int>(taskData->inputs_count[1]); i++) {
      limits.push_back(value_2[i]);
    }
  }

  return true;
}

bool frolova_e_Simpson_method_mpi::SimpsonmethodParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    int* value = reinterpret_cast<int*>(taskData->inputs[0]);
    if (taskData->inputs_count[0] != 3) {
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
  }
  return true;
}

bool frolova_e_Simpson_method_mpi::SimpsonmethodParallel::run() {
  internal_order_test();

  broadcast(world, functionid, 0);
  broadcast(world, dimension, 0);

  if (world.rank() == 0) {
    localdivisions = divisions / world.size();
    if (localdivisions % 2 != 0) {
      localdivisions++;
    }
  }
  broadcast(world, localdivisions, 0);
  broadcast(world, limits, 0);

  if (world.rank() == 0) {
    size_t size = world.size();

    for (size_t i = 0; i < size; i++) {
      std::vector<double> loclim;

      for (size_t j = 0; j < dimension; j++) {
        double a = limits[2 * j];
        double b = limits[2 * j + 1];
        double step = (b - a) / size;

        double lim1;
        double lim2;

        if (j == 0) {
          lim1 = a + i * step;
          if (i < size - 1) {
            lim2 = a + (i + 1) * step;
          } else {
            lim2 = b;
          }
        } else {
          lim1 = a;
          lim2 = b;
        }

        if (i == 0) {
          localLimits.push_back(lim1);
          localLimits.push_back(lim2);
        } else {
          loclim.push_back(lim1);
          loclim.push_back(lim2);
        }
      }

      if (i != 0) {
        world.send(i, 0, loclim);
      }
    }
  }

  if (world.rank() != 0) {
    world.recv(0, 0, localLimits);
  }

  func = frolova_e_Simpson_method_mpi::functionRegistry[functionid];

  localres = frolova_e_Simpson_method_mpi::Simpson_Method(func, localdivisions, dimension, localLimits);

  reduce(world, localres, resIntegral, std::plus<>(), 0);

  return true;
}

bool frolova_e_Simpson_method_mpi::SimpsonmethodParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = resIntegral;
  }

  return true;
}