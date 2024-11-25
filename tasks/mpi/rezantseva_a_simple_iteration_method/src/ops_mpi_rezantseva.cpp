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

  size_t n = 0;
  size_t delta = 0;
  size_t remainder = 0;

  if (world.rank() == 0) {
    n = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    num_processes_ = world.size();
    delta = n / num_processes_;      // Calculate base size for each process
    remainder = n % num_processes_;  // Calculate remaining elements
  }
  boost::mpi::broadcast(world, num_processes_, 0);
  boost::mpi::broadcast(world, n, 0);
  counts_.resize(num_processes_);  // Vector to store counts for each process

  if (world.rank() == 0) {
    // Distribute sizes to each process
    for (unsigned int i = 0; i < num_processes_; ++i) {
      counts_[i] = delta + (i < remainder ? 1 : 0);  // Assign 1 additional element to the first 'remainder' processes
    }
  }
  boost::mpi::broadcast(world, counts_.data(), num_processes_, 0);

  if (world.rank() == 0) {
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
  } else {
    x_.resize(n);
    prev_x_.resize(n);
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI::run() {
  internal_order_test();
  size_t n = x_.size();
  size_t iteration = 0;

  std::vector<double> local_A(counts_[world.rank()] * n);
  std::vector<double> local_b(counts_[world.rank()]);
  std::vector<double> local_x(counts_[world.rank()], 0.0);

  if (world.rank() == 0) {
    for (size_t i = 0; i < counts_[0]; i++) {
      std::copy(A_.begin() + i * n, A_.begin() + (i + 1) * n, local_A.begin() + i * n);
    }
    std::copy(b_.begin(), b_.begin() + counts_[0], local_b.begin());

    // send parts to other proc
    size_t offset = counts_[0];
    for (size_t proc = 1; proc < num_processes_; proc++) {
      std::vector<double> proc_A(counts_[proc] * n);
      for (size_t i = 0; i < counts_[proc]; i++) {
        std::copy(A_.begin() + (offset + i) * n, A_.begin() + (offset + i + 1) * n, proc_A.begin() + i * n);
      }
      world.send(proc, 0, proc_A);

      std::vector<double> proc_b(counts_[proc]);
      std::copy(b_.begin() + offset, b_.begin() + offset + counts_[proc], proc_b.begin());
      world.send(proc, 1, proc_b);

      offset += counts_[proc];
    }
  } else {
    world.recv(0, 0, local_A);
    world.recv(0, 1, local_b);
  }

  while (iteration < maxIteration_) {
    // local offset
    size_t offset = 0;
    for (size_t i = 0; i < world.rank(); i++) {
      offset += counts_[i];
    }

    // calculate
    for (size_t i = 0; i < counts_[world.rank()]; i++) {
      double sum = 0.0;
      for (size_t j = 0; j < n; j++) {
        if (j != (offset + i)) {
          sum += local_A[i * n + j] * x_[j];
        }
      }
      local_x[i] = (local_b[i] - sum) / local_A[i * n + (offset + i)];
    }

    if (world.rank() == 0) {
      // save results which one was calculated on proc 1
      prev_x_ = x_;
      std::copy(local_x.begin(), local_x.end(), x_.begin());

      // get results from another proc
      size_t gather_offset = counts_[0];
      for (size_t proc = 1; proc < num_processes_; proc++) {
        std::vector<double> proc_x(counts_[proc]);
        world.recv(proc, 2, proc_x);
        std::copy(proc_x.begin(), proc_x.end(), x_.begin() + gather_offset);
        gather_offset += counts_[proc];
      }

      bool converged = isTimeToStop(prev_x_, x_);
      if (converged) {
        // send check result
        for (size_t proc = 1; proc < num_processes_; proc++) {
          world.send(proc, 3, x_);
          world.send(proc, 4, &converged, 1);
        }
        break;
      } else {
        // send new approach
        for (size_t proc = 1; proc < num_processes_; proc++) {
          world.send(proc, 3, x_);
          world.send(proc, 4, &converged, 1);
        }
      }
    } else {
      // non zero proc sends their results
      world.send(0, 2, local_x);

      // get new approach
      world.recv(0, 3, x_);
      bool converged;
      world.recv(0, 4, &converged, 1);
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