// Copyright 2024 Lupsha Egor
#include "mpi/lupsha_e_rect_integration/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

void lupsha_e_rect_integration_mpi::TestMPITaskSequential::function_set(const std::function<double(double)>& func) {
  f = func;
}

void lupsha_e_rect_integration_mpi::TestMPITaskParallel::function_set(const std::function<double(double)>& func) {
  f = func;
}

bool lupsha_e_rect_integration_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  lower_bound = *reinterpret_cast<double*>(taskData->inputs[0]);
  upper_bound = *reinterpret_cast<double*>(taskData->inputs[1]);
  num_intervals = *reinterpret_cast<int*>(taskData->inputs[2]);

  results_.resize(1, 0.0);

  return true;
}

bool lupsha_e_rect_integration_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs.size() < 3) {
    return false;
    std::cout << "Validation failed: not enough input data." << std::endl;
  }

  double validation_lower_bound = *reinterpret_cast<double*>(taskData->inputs[0]);
  double validation_upper_bound = *reinterpret_cast<double*>(taskData->inputs[1]);
  if (validation_lower_bound >= validation_upper_bound) {
    std::cout << "Validation failed: lower_bound >= upper_bound." << std::endl;
    return false;
  }

  int validation_num_intervals = *reinterpret_cast<int*>(taskData->inputs[2]);
  if (validation_num_intervals <= 0) {
    std::cout << "Validation failed: num_intervals <= 0." << std::endl;
    return false;
  }

  if (!f) {
    std::cout << "Validation failed: function is not set." << std::endl;
    return false;
  }

  return true;
}

bool lupsha_e_rect_integration_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  double width = (upper_bound - lower_bound) / num_intervals;
  input_.resize(num_intervals);
  double sum = 0.0;

  for (int i = 0; i < num_intervals; ++i) {
    double x = lower_bound + i * width;
    sum += f(x) * width;
  }
  results_[0] = sum;

  return true;
}

bool lupsha_e_rect_integration_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  *reinterpret_cast<double*>(taskData->outputs[0]) = results_[0];

  return true;
}

double lupsha_e_rect_integration_mpi::TestMPITaskParallel::integrate(const std::function<double(double)>& fun,
                                                                     double lower_bound_, double upper_bound_,
                                                                     int num_intervals_) {
  int rank = world.rank();
  int size = world.size();

  double width = (upper_bound_ - lower_bound_) / num_intervals_;
  int local_num_intervals = num_intervals_ / size;
  int remainder = num_intervals_ % size;

  if (rank < remainder) {
    local_num_intervals = local_num_intervals + 1;
  }

  double local_start = lower_bound_ + rank * local_num_intervals * width;

  double local_sum = 0.0;
  for (int i = 0; i < local_num_intervals; ++i) {
    double x = local_start + i * width;
    local_sum += fun(x) * width;
  }

  return local_sum;
}

bool lupsha_e_rect_integration_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  unsigned int d = 0;
  if (world.rank() == 0) {
    d = num_intervals / world.size();
  }
  MPI_Bcast(&d, 1, MPI_UNSIGNED, 0, world);

  if (world.rank() == 0) {
    lower_bound = *reinterpret_cast<double*>(taskData->inputs[0]);
    upper_bound = *reinterpret_cast<double*>(taskData->inputs[1]);
    num_intervals = *reinterpret_cast<int*>(taskData->inputs[2]);
  }

  MPI_Bcast(&lower_bound, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&upper_bound, 1, MPI_DOUBLE, 0, world);
  MPI_Bcast(&num_intervals, 1, MPI_INT, 0, world);

  return true;
}

bool lupsha_e_rect_integration_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.size() < 3) {
      std::cout << "Validation failed: not enough input data." << std::endl;
      return false;
    }

    double validation_lower_bound = *reinterpret_cast<double*>(taskData->inputs[0]);
    double validation_upper_bound = *reinterpret_cast<double*>(taskData->inputs[1]);
    if (validation_lower_bound >= validation_upper_bound) {
      std::cout << "Validation failed: lower_bound >= upper_bound." << std::endl;
      return false;
    }

    int validation_num_intervals = *reinterpret_cast<int*>(taskData->inputs[2]);
    if (validation_num_intervals <= 0) {
      std::cout << "Validation failed: num_intervals <= 0." << std::endl;
      return false;
    }

    if (!f) {
      std::cout << "Validation failed: function is not set." << std::endl;
      return false;
    }
  }

  return true;
}

bool lupsha_e_rect_integration_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  local_sum_ = integrate(f, lower_bound, upper_bound, num_intervals);
  MPI_Reduce(&local_sum_, &global_sum_, 1, MPI_DOUBLE, MPI_SUM, 0, world);
  return true;
}

bool lupsha_e_rect_integration_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_sum_;
  }

  return true;
}