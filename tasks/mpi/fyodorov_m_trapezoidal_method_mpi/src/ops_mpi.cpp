// Copyright 2023 Nesterov Alexander
#include "mpi/fyodorov_m_trapezoidal_method_mpi/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
// #include <boost/serialization/access.hpp>
// #include <boost/serialization/vector.hpp>
#include <functional>
#include <string>
#include <thread>
#include <vector>

bool fyodorov_m_trapezoidal_method_mpi::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  if (taskData->inputs_count.size() < 4) return false;  //

  func_ = *reinterpret_cast<std::function<double(const std::vector<double>&)>*>(taskData->inputs[0]);

  size_t dim = taskData->inputs_count[1];
  lower_bounds_.resize(dim);
  upper_bounds_.resize(dim);
  intervals_.resize(dim);

  for (size_t i = 0; i < dim; ++i) {
    lower_bounds_[i] = reinterpret_cast<double*>(taskData->inputs[1])[i];
    upper_bounds_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
    intervals_[i] = reinterpret_cast<int*>(taskData->inputs[3])[i];
  }
  result_ = 0.0;
  return true;
}

bool fyodorov_m_trapezoidal_method_mpi::TestTaskSequential::validation() {
  internal_order_test();

  if (taskData->outputs_count.size() != 1 || taskData->outputs_count[0] != 1) {
    return false;
  }

  // Проверка на корректные границы интегрирования
  for (size_t i = 0; i < lower_bounds_.size(); ++i) {
    if (lower_bounds_[i] >= upper_bounds_[i]) {
      return false;
    }
  }

  // Проверка на корректные интервалы
  for (size_t i = 0; i < intervals_.size(); ++i) {
    if (intervals_[i] <= 0) {
      return false;
    }
  }

  return true;
}

bool fyodorov_m_trapezoidal_method_mpi::TestTaskSequential::run() {
  internal_order_test();

  size_t dim = lower_bounds_.size();
  std::vector<double> step(dim);
  for (size_t i = 0; i < dim; ++i) step[i] = (upper_bounds_[i] - lower_bounds_[i]) / intervals_[i];

  std::function<double(std::vector<double>, size_t)> integrate = [&](std::vector<double> current_point,
                                                                     size_t current_dim) -> double {
    if (current_dim == dim) {
      return func_(current_point);
    }
    double sum = 0.0;
    for (int i = 0; i <= intervals_[current_dim]; ++i) {
      current_point[current_dim] = lower_bounds_[current_dim] + i * step[current_dim];
      double value;
      if (i == 0 || i == intervals_[current_dim]) {
        value = integrate(current_point, current_dim + 1);
      } else {
        value = 2 * integrate(current_point, current_dim + 1);
      }
      sum += value;
    }
    return sum * step[current_dim] / 2.0;
  };
  std::vector<double> start_point(dim, 0.0);
  result_ = integrate(start_point, 0);
  return true;
}

bool fyodorov_m_trapezoidal_method_mpi::TestTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  return true;
}

// Implementation for parallel task

bool fyodorov_m_trapezoidal_method_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  world = boost::mpi::communicator();
  // Init value for input and output
  if (world.rank() == 0) {
    if (taskData->inputs_count.size() < 3) return false;  //

    size_t dim = taskData->inputs_count[1];
    lower_bounds_.resize(dim);
    upper_bounds_.resize(dim);
    intervals_.resize(dim);

    for (size_t i = 0; i < dim; ++i) {
      lower_bounds_[i] = reinterpret_cast<double*>(taskData->inputs[0])[i];
      upper_bounds_[i] = reinterpret_cast<double*>(taskData->inputs[1])[i];
      intervals_[i] = reinterpret_cast<int*>(taskData->inputs[2])[i];
    }

    result_ = 0.0;
  }
  return true;
}

bool fyodorov_m_trapezoidal_method_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  for (size_t i = 0; i < lower_bounds_.size(); ++i) {
    if (lower_bounds_[i] >= upper_bounds_[i]) {
      return false;
    }
  }

  for (size_t i = 0; i < intervals_.size(); ++i) {
    if (intervals_[i] <= 0) {
      return false;
    }
  }

  if (world.rank() == 0) {
    return taskData->outputs_count[0] == 1;
  }

  return true;
}

bool fyodorov_m_trapezoidal_method_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  size_t dim = (world.rank() == 0) ? lower_bounds_.size() : 0;
  boost::mpi::broadcast(world, dim, 0);
  boost::mpi::broadcast(world, intervals_, 0);
  boost::mpi::broadcast(world, lower_bounds_, 0);
  boost::mpi::broadcast(world, upper_bounds_, 0);

  std::vector<double> step(dim);
  for (size_t i = 0; i < dim; i++) step[i] = (upper_bounds_[i] - lower_bounds_[i]) / intervals_[i];

  if (dim > 0) {
    int interval_per_process = intervals_[0] / world.size();
    int remainder = intervals_[0] % world.size();

    intervals_[0] = interval_per_process + (world.rank() < remainder ? 1 : 0);
    double interval_step = (upper_bounds_[0] - lower_bounds_[0]) / world.size();
    lower_bounds_[0] = world.rank() * interval_step;
    upper_bounds_[0] = (world.rank() + 1) * interval_step;
  }

  std::function<double(std::vector<double>, size_t)> integrate = [&](std::vector<double> current_point,
                                                                     size_t current_dim) -> double {
    if (current_dim == dim) {
      return func_(current_point);
    }
    double sum = 0.0;
    for (int i = 0; i <= intervals_[current_dim]; ++i) {
      current_point[current_dim] = lower_bounds_[current_dim] + i * step[current_dim];
      double value;
      if (i == 0 || i == intervals_[current_dim]) {
        value = integrate(current_point, current_dim + 1);
      } else {
        value = 2 * integrate(current_point, current_dim + 1);
      }
      sum += value;
    }
    return sum * step[current_dim] / 2.0;
  };

  std::vector<double> start_point = lower_bounds_;
  double local_result = 0;

  for (int i = 0; i <= intervals_[0]; i++) {
    start_point[0] = lower_bounds_[0] + i * step[0];
    double value;
    if (i == 0 || i == intervals_[0]) {
      value = integrate(start_point, 1);
    } else {
      value = 2 * integrate(start_point, 1);
    }
    local_result += value;
  }

  double global_result = 0.0;
  boost::mpi::reduce(world, local_result, global_result, std::plus<>(), 0);

  if (world.rank() == 0) {
    result_ = global_result * step[0] / 2.0;
  }

  return true;
}

bool fyodorov_m_trapezoidal_method_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  }
  return true;
}
