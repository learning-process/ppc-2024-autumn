// Copyright 2024 Ivanov Mike
#include "mpi/ivanov_m_integration_trapezoid/include/ops_mpi.hpp"

bool ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  auto* input = reinterpret_cast<double*>(taskData->inputs[0]);
  a_ = input[0];
  b_ = input[1];
  n_ = static_cast<int>(input[2]);
  result_ = 0.0;

  return true;
}

bool ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == 1;
}

bool ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  if (a_ == b_) return true;
  double step_ = (b_ - a_) / n_;
  for (int i = 0; i < n_; i++) result_ += (f_(a_ + i * step_) + f_(a_ + (i + 1) * step_));
  result_ = result_ / 2 * step_;
  return true;
}

bool ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  return true;
}

void ivanov_m_integration_trapezoid_mpi::TestMPITaskSequential::add_function(const std::function<double(double)>& f) {
  f_ = f;
}

bool ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == 1;
  }
  return true;
}

bool ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* input = reinterpret_cast<double*>(taskData->inputs[0]);
    a_ = input[0];
    b_ = input[1];
    n_ = static_cast<int>(input[2]);
    result_ = 0.0;
  }
  return true;
}

bool ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int rank = world.rank();
  int size = world.size();
  double step;
  double local_result = 0.0;
  if (rank == 0) {
    step = (b_ - a_) / n_;
  }
  broadcast(world, a_, 0);
  broadcast(world, b_, 0);
  broadcast(world, n_, 0);
  broadcast(world, step, 0);

  if (a_ == b_) return true;

  for (int i = rank; i < n_; i += size) local_result += (f_(a_ + i * step) + f_(a_ + (i + 1) * step));
  reduce(world, local_result, result_, std::plus<>(), 0);

  if (rank == 0) result_ = result_ / 2 * step;

  return true;
}

bool ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = result_;
  }
  return true;
}

void ivanov_m_integration_trapezoid_mpi::TestMPITaskParallel::add_function(const std::function<double(double)>& f) {
  f_ = f;
}