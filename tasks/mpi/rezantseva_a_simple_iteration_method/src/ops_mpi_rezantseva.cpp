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

  size_t delta = 0;
  size_t remainder = 0;

  if (world.rank() == 0) {
    n_ = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    num_processes_ = world.size();

    // inicialized
    A_.assign(n_ * n_, 0.0);
    b_.assign(n_, 0.0);
    x_.assign(n_, 0.0);
    prev_x_.assign(n_, 0.0);

    // fill matrix A and vector b
    for (size_t i = 0; i < n_; ++i) {    // row
      for (size_t j = 0; j < n_; ++j) {  // column
        A_[i * n_ + j] = reinterpret_cast<double*>(taskData->inputs[1])[i * n_ + j];
      }
      b_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
    }
    // calculate offset
    counts_.resize(num_processes_);
    delta = n_ / num_processes_;
    remainder = n_ % num_processes_;

    for (size_t i = 0; i < num_processes_; ++i) {
      counts_[i] = delta + (i < remainder ? 1 : 0);  // Assign 1 additional row to the first 'remainder' processes
    }
  }
  boost::mpi::broadcast(world, n_, 0);
  boost::mpi::broadcast(world, num_processes_, 0);

  x_.resize(n_);
  prev_x_.resize(n_);
  counts_.resize(num_processes_);

  boost::mpi::broadcast(world, counts_.data(), num_processes_, 0);

  return true;
}

// working with small data
bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI::run() {
  internal_order_test();
  size_t iteration = 0;
  std::vector<int> offsets_x(num_processes_, 0);

  std::vector<double> local_A(counts_[world.rank()] * n_);
  std::vector<double> local_b(counts_[world.rank()]);
  std::vector<double> local_x(n_, 0.0);
  std::vector<double> res_x(counts_[world.rank()]);

  // send data
  if (world.rank() == 0) {
    size_t offset_remainder_A = counts_[0] * n_;
    size_t offset_remainder_b = counts_[0];
    const size_t chunk_size = 100;

    for (size_t proc = 1; proc < num_processes_; proc++) {
      size_t current_count = counts_[proc];
      offsets_x[proc] = offsets_x[proc - 1] + counts_[proc - 1];

      world.send(proc, 1, b_.data() + offset_remainder_b, current_count);
      world.send(proc, 2, x_.data(), n_);
      // send A in parts
      size_t total_elements = current_count * n_;
      size_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;

      world.send(proc, 3, &num_chunks, 1);  // send counts of parts
      for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        size_t start = chunk * chunk_size;
        size_t size = std::min(chunk_size, total_elements - start);
        world.send(proc, 4, A_.data() + offset_remainder_A + start, size);
      }

      offset_remainder_b += current_count;
      offset_remainder_A += current_count * n_;
    }
  }
  boost::mpi::broadcast(world, offsets_x.data(), num_processes_, 0);

  // get data
  if (world.rank() > 0) {
    world.recv(0, 1, local_b.data(), counts_[world.rank()]);
    world.recv(0, 2, local_x.data(), n_);
    // get parts of  A
    size_t num_chunks;
    world.recv(0, 3, &num_chunks, 1);  // get count of parts

    size_t received = 0;
    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
      size_t remaining = (counts_[world.rank()] * n_) - received;
      size_t size = std::min(size_t(100), remaining);
      world.recv(0, 4, local_A.data() + received, size);
      received += size;
    }
  } else {
    local_b.assign(b_.begin(), b_.begin() + counts_[0]);
    local_A.assign(A_.begin(), A_.begin() + counts_[0] * n_);
    local_x = x_;
  }

  bool should_stop = false;
  // method
  while (iteration < maxIteration_) {
    if (world.rank() == 0) {
      prev_x_ = x_;
    }
    //  local offset
    size_t offset = 0;
    for (size_t i = 0; i < static_cast<size_t>(world.rank()); i++) {
      offset += counts_[i] * n_;
    }
    // calculate
    for (size_t i = 0; i < counts_[world.rank()]; i++) {
      double sum = 0.0;
      size_t global_i = offsets_x[world.rank()] + i;  // global id row
      for (size_t j = 0; j < n_; j++) {
        if (j != (global_i)) {
          sum += local_A[i * n_ + j] * local_x[j];
        }
      }
      double diag = local_A[i * n_ + global_i];
      res_x[i] = (local_b[i] - sum) / diag;
    }
    // get new local approach on rank 0
    boost::mpi::gatherv(world, res_x, x_.data(), counts_, offsets_x, 0);
    // check
    if (world.rank() == 0) {
      should_stop = isTimeToStop(prev_x_, x_);
    }
    boost::mpi::broadcast(world, should_stop, 0);
    if (should_stop) {
      break;
    }
    if (world.rank() == 0) {
      prev_x_ = x_;
    }
    boost::mpi::broadcast(world, x_.data(), n_, 0);
    local_x = x_;

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