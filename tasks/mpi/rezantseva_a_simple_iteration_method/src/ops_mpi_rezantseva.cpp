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
  for (size_t i = 0; i < n_; ++i) {  // row

    double Aii = std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n_ + i]);
    double sum = 0.0;

    for (size_t j = 0; j < n_; ++j) {  // column
      if (i != j) {
        sum += std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n_ + j]);
      }
    }
    if (Aii <= sum) {
      return false;
    }
  }
  return true;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential::checkMatrixNorm() {
  double max_row_sum = 0.0;
  for (size_t i = 0; i < n_; ++i) {
    double row_sum = 0.0;
    for (size_t j = 0; j < n_; ++j) {
      row_sum += std::abs(B_[i * n_ + j]);
    }
    max_row_sum = std::max(max_row_sum, row_sum);
  }
  return max_row_sum < 1.0;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential::validation() {
  internal_order_test();
  n_ = *reinterpret_cast<size_t*>(taskData->inputs[0]);
  return (taskData->inputs_count.size() == 3) && (taskData->outputs_count.size() == 1) && (n_ > 0) && checkMatrix();
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential::pre_processing() {
  internal_order_test();
  A_.assign(n_ * n_, 0.0);
  b_.assign(n_, 0.0);
  x_.assign(n_, 0.0);

  B_.assign(n_ * n_, 0.0);
  c_.assign(n_, 0.0);
  // fill matrix A and vector b
  for (size_t i = 0; i < n_; ++i) {    // row
    for (size_t j = 0; j < n_; ++j) {  // column
      A_[i * n_ + j] = reinterpret_cast<double*>(taskData->inputs[1])[i * n_ + j];
    }
    b_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
  }
  // fill transition matrix B and iteration vector c
  for (size_t i = 0; i < n_; ++i) {
    double diag = A_[i * n_ + i];
    for (size_t j = 0; j < n_; ++j) {
      if (i != j) {                               // diagonal elements of B remain zero
        B_[i * n_ + j] = -A_[i * n_ + j] / diag;  // Bij = -Aij/Aii
      }
    }
    c_[i] = b_[i] / diag;  // ci = bi/Aii
  }

  return true;
}

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationSequential::run() {
  internal_order_test();
  size_t iteration = 0;
  std::vector<double> prev_x(n_, 0.0);

  while (iteration < maxIteration_) {
    std::copy(x_.begin(), x_.end(), prev_x.begin());  // saved previous approach

    // new approach x = Bx + c
    for (size_t i = 0; i < n_; i++) {
      x_[i] = c_[i];
      for (size_t j = 0; j < n_; j++) {
        x_[i] += B_[i * n_ + j] * prev_x[j];
      }
    }
    if (isTimeToStop(prev_x, x_)) {
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
    for (size_t i = 0; i < n_; ++i) {  // row

      double Aii = std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n_ + i]);
      double sum = 0.0;

      for (size_t j = 0; j < n_; ++j) {  // column
        if (i != j) {
          sum += std::fabs(reinterpret_cast<double*>(taskData->inputs[1])[i * n_ + j]);
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
    n_ = *reinterpret_cast<size_t*>(taskData->inputs[0]);
    return (taskData->inputs_count.size() == 3) && (taskData->outputs_count.size() == 1) && (n_ > 0) && checkMatrix();
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

    B_.assign(n_ * n_, 0.0);
    c_.assign(n_, 0.0);

    // fill matrix A and vector b
    for (size_t i = 0; i < n_; ++i) {    // row
      for (size_t j = 0; j < n_; ++j) {  // column
        A_[i * n_ + j] = reinterpret_cast<double*>(taskData->inputs[1])[i * n_ + j];
      }
      b_[i] = reinterpret_cast<double*>(taskData->inputs[2])[i];
    }

    // fill transition matrix B and iteration vector c
    for (size_t i = 0; i < n_; ++i) {
      double diag = A_[i * n_ + i];
      for (size_t j = 0; j < n_; ++j) {
        if (i != j) {                               // diagonal elements of B remain zero
          B_[i * n_ + j] = -A_[i * n_ + j] / diag;  // Bij = -Aij/Aii
        }
      }
      c_[i] = b_[i] / diag;  // ci = bi/Aii
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

bool rezantseva_a_simple_iteration_method_mpi::SimpleIterationMPI::run() {
  internal_order_test();
  size_t iteration = 0;
  std::vector<int> offsets_x(num_processes_, 0);

  std::vector<double> local_B(counts_[world.rank()] * n_);
  std::vector<double> local_c(counts_[world.rank()]);
  std::vector<double> local_x(n_, 0.0);
  std::vector<double> res_x(counts_[world.rank()]);
  const size_t chunk_size = 100;
  // send data
  if (world.rank() == 0) {
    size_t offset_remainder_B = counts_[0] * n_;
    size_t offset_remainder_c = counts_[0];

    for (size_t proc = 1; proc < num_processes_; proc++) {
      size_t current_count = counts_[proc];
      offsets_x[proc] = offsets_x[proc - 1] + counts_[proc - 1];

      world.send(proc, 1, c_.data() + offset_remainder_c, current_count);
      world.send(proc, 2, x_.data(), n_);
      // send B in parts
      size_t total_elements = current_count * n_;
      size_t num_chunks = (total_elements + chunk_size - 1) / chunk_size;

      world.send(proc, 3, &num_chunks, 1);  // send counts of parts
      for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
        size_t start = chunk * chunk_size;
        size_t size = std::min(chunk_size, total_elements - start);
        world.send(proc, 4, B_.data() + offset_remainder_B + start, size);
      }

      offset_remainder_c += current_count;
      offset_remainder_B += current_count * n_;
    }
  }
  boost::mpi::broadcast(world, offsets_x.data(), num_processes_, 0);

  // get data
  if (world.rank() > 0) {
    world.recv(0, 1, local_c.data(), counts_[world.rank()]);
    world.recv(0, 2, local_x.data(), n_);
    // get parts of  B
    size_t num_chunks;
    world.recv(0, 3, &num_chunks, 1);  // get count of parts

    size_t received = 0;
    for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
      size_t remaining = (counts_[world.rank()] * n_) - received;
      size_t size = std::min(size_t(chunk_size), remaining);
      world.recv(0, 4, local_B.data() + received, size);
      received += size;
    }
  } else {
    local_c.assign(c_.begin(), c_.begin() + counts_[0]);
    local_B.assign(B_.begin(), B_.begin() + counts_[0] * n_);
    local_x = x_;
  }

  bool should_stop = false;
  // method
  while (iteration < maxIteration_) {
    if (world.rank() == 0) {
      prev_x_ = x_;
    }
    // calculate new approach x = Bx + c
    for (size_t i = 0; i < static_cast<size_t>(counts_[world.rank()]); i++) {
      res_x[i] = local_c[i];
      for (size_t j = 0; j < n_; j++) {
        res_x[i] += local_B[i * n_ + j] * local_x[j];
      }
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