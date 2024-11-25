// Copyright 2024 Nesterov Alexander
#include "mpi/rezantseva_a_simple_iteration_method/include/ops_mpi_rezantseva.hpp"

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential::isTimeToStop(
    const std::vector<double>& x0, const std::vector<double>& x1) const {
  double max_precision = 0.0;  // max precision between iterations

  for (size_t k = 0; k < x0.size(); k++) {
    double precision = std::fabs(x1[k] - x0[k]);  // |x1^(i+1) - x1^i|
    if (precision > max_precision) {
      max_precision = precision;
    }
  }
  return (max_precision < epsilon_);
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential::checkMatrix() {
  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);

  for (size_t i = 0; i < n; ++i) {  // row

    double Aii = std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + i]);
    double sum = 0.0;

    for (size_t j = 0; j < n; ++j) {  // column
      if (i != j) {
        sum += std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + j]);
      }
    }
    if (Aii <= sum) {
      return false;
    }
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential::validation() {
  internal_order_test();
  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  return (taskData->inputs_count.size() == 3) && (taskData->outputs_count.size() == 1) && (n > 0) &&
         (checkMatrix() == true);
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential::pre_processing() {
  internal_order_test();
  size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  A_.assign(n * n, 0.0);
  b_.assign(n, 0.0);
  x_.assign(n, 0.0);
  // fill matrix A and vector b
  for (size_t i = 0; i < n; ++i) {    // row
    for (size_t j = 0; j < n; ++j) {  // column
      A_[i * n + j] = reinterpret_cast<double*>(taskData->inputs[1])[i * n + j];
    }
    b_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential::run() {
  internal_order_test();
  size_t iteration = 0;
  size_t n = b_.size();
  std::vector<double> x0(n, 0.0);

  while (iteration < maxIteration_) {
    std::copy(x_.begin(), x_.end(), x0.begin());  // move previous decisions to vec x0
    for (size_t i = 0; i < n; i++) {
      double sum = 0;
      for (size_t j = 0; j < n; j++) {
        if (j != i) {
          sum += A_[i * n + j] * x0[j];  // example: A12*x2 + A13*x3+..+ A1n*xn
        }
      }
      x_[i] = (b_[i] - sum) / A_[i * n + i];
    }
    if (isTimeToStop(x0, x_)) {
      break;
    }
    iteration++;
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < x_.size(); ++i) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x_[i];
  }
  return true;
}

// MPI

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI::isTimeToStop(const std::vector<double>& x0,
                                                                                const std::vector<double>& x1) const {
  double max_precision = 0.0;  // max precision between iterations

  for (size_t k = 0; k < x0.size(); k++) {
    double precision = std::fabs(x1[k] - x0[k]);  // |x1^(i+1) - x1^i|
    if (precision > max_precision) {
      max_precision = precision;
    }
  }
  return (max_precision < epsilon_);
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI::checkMatrix() {
  if (world.rank() == 0) {
    size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);

    for (size_t i = 0; i < n; ++i) {  // row

      double Aii = std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + i]);
      double sum = 0.0;

      for (size_t j = 0; j < n; ++j) {  // column
        if (i != j) {
          sum += std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n + j]);
        }
      }
      if (Aii <= sum) {
        return false;
      }
    }
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    return (taskData->inputs_count.size() == 3) && (taskData->outputs_count.size() == 1) && (n > 0) &&
           (checkMatrix() == true);
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI::pre_processing() {
  internal_order_test();

  size_t total_elements = 0;  // count of elements = n
  size_t delta = 0;
  size_t remainder = 0;

  if (world.rank() == 0) {
    total_elements = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    num_processes_ = world.size();
    delta = total_elements / num_processes_;      // Calculate base size for each process
    remainder = total_elements % num_processes_;  // Calculate remaining elements
  }
  boost::mpi::broadcast(world, num_processes_, 0);

  counts_.resize(num_processes_);  // Vector to store counts for each process

  if (world.rank() == 0) {
    // Distribute sizes to each process
    for (unsigned int i = 0; i < num_processes_; ++i) {
      counts_[i] = delta + (i < remainder ? 1 : 0);  // Assign 1 additional element to the first 'remainder' processes
    }
  }
  boost::mpi::broadcast(world, counts_.data(), num_processes_, 0);

  if (world.rank() == 0) {
    size_t n = *reinterpret_cast<size_t*>(taskData->inputs[0]);

    A_.assign(n * n, 0.0);
    b_.assign(n, 0.0);
    x_.assign(n, 0.0);
    prev_x_.assign(n, 0.0);

    // fill matrix A and vector b
    for (size_t i = 0; i < n; ++i) {    // row
      for (size_t j = 0; j < n; ++j) {  // column
        A_[i * n + j] = reinterpret_cast<double*>(taskData->inputs[1])[i * n + j];
      }
      b_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
    }
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI::run() {
  internal_order_test();
  size_t n = 0;
  size_t iteration = 0;
  if (world.rank() == 0) {
    n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  }
  boost::mpi::broadcast(world, n, 0);

  x_.resize(n, 0.0);
  prev_x_.resize(n, 0.0);

  if (world.rank() == 0) {
    size_t offset_remainder = counts_[0];
    for (size_t proc = 1; proc < num_processes_; proc++) {
      size_t current_count = counts_[proc];

      // send part of A as vec
      std::vector<double> temp_A;
      for (size_t i = 0; i < current_count; i++) {
        temp_A.insert(temp_A.end(), A_.begin() + (offset_remainder + i) * n,
                      A_.begin() + (offset_remainder + i + 1) * n);
      }
      world.send(proc, 0, temp_A);

      // send part of vec b
      std::vector<double> temp_b(b_.begin() + offset_remainder, b_.begin() + offset_remainder + current_count);
      world.send(proc, 1, temp_b);

      offset_remainder += current_count;
    }
  }

  std::vector<double> local_A(counts_[world.rank()] * n);
  std::vector<double> local_b(counts_[world.rank()]);
  std::vector<double> local_x(counts_[world.rank()], 0.0);

  if (world.rank() > 0) {
    world.recv(0, 0, local_A);
    world.recv(0, 1, local_b);
  } else {
    // copy local data for proc 0
    for (size_t i = 0; i < counts_[0]; i++) {
      std::copy(A_.begin() + i * n, A_.begin() + (i + 1) * n, local_A.begin() + i * n);
    }
    std::copy(b_.begin(), b_.begin() + counts_[0], local_b.begin());
  }

  // Simple Iteration Method
  while (iteration < maxIteration_) {
    boost::mpi::broadcast(world, x_.data(), n, 0);

    size_t row_offset = 0;
    for (size_t i = 0; i < static_cast<size_t>(world.rank()); i++) {
      row_offset += counts_[i];
    }

    for (size_t i = 0; i < counts_[world.rank()]; i++) {
      double sum = 0.0;
      for (size_t j = 0; j < n; j++) {
        if (j != (row_offset + i)) {
          sum += local_A[i * n + j] * x_[j];
        }
      }
      local_x[i] = (local_b[i] - sum) / local_A[i * n + (row_offset + i)];
    }

    if (world.rank() == 0) {
      prev_x_ = x_;
      std::copy(local_x.begin(), local_x.end(), x_.begin());

      size_t offset = counts_[0];
      for (size_t proc = 1; proc < num_processes_; proc++) {
        std::vector<double> temp_x;
        world.recv(proc, 2, temp_x);
        std::copy(temp_x.begin(), temp_x.end(), x_.begin() + offset);
        offset += counts_[proc];
      }

      bool converged = isTimeToStop(prev_x_, x_);
      boost::mpi::broadcast(world, converged, 0);
      if (converged) break;
    } else {
      world.send(0, 2, local_x);
      bool converged;
      boost::mpi::broadcast(world, converged, 0);
      if (converged) break;
    }
    iteration++;
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < x_.size(); ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = x_[i];
    }
  }
  return true;
}
