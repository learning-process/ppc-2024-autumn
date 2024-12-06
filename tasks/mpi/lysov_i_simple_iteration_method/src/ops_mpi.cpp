// Copyright 2023 Nesterov Alexander
#include "mpi/lysov_i_simple_iteration_method/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTask::isDiagonallyDominant() {
  for (int i = 0; i < input_size_; ++i) {
    double diagonal = std::abs(A_[i][i]);
    double sum = 0.0;

    for (int j = 0; j < input_size_; ++j) {
      if (i != j) {
        sum += std::abs(A_[i][j]);
      }
    }

    if (diagonal <= sum) {
      return false;
    }
  }

  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTask::transformSystem() {
  B_.resize(input_size_, std::vector<double>(input_size_, 0.0));
  g_.resize(input_size_);

  for (int i = 0; i < input_size_; ++i) {
    double diagonal = A_[i][i];

    if (diagonal == 0.0) {
      std::cerr << "Zero diagonal element detected, cannot apply iteration.";
      return false;
    }

    g_[i] = b_[i] / diagonal;

    for (int j = 0; j < input_size_; ++j) {
      if (i != j) {
        B_[i][j] = -A_[i][j] / diagonal;
      }
    }
  }
  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTask::pre_processing() {
  internal_order_test();

  auto* X_raw = reinterpret_cast<double*>(taskData->outputs[0]);
  x_.resize(input_size_);
  for (int i = 0; i < input_size_; ++i) {
    x_[i] = X_raw[i];
  }

  tolerance_ = 1e-6;

  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTask::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] != taskData->inputs_count[1] || taskData->inputs_count[0] != taskData->outputs_count[0])
    return false;
  input_size_ = taskData->inputs_count[0];
  if (input_size_ <= 0) {
    return false;
  }
  A_.resize(input_size_, std::vector<double>(input_size_));
  auto* A_raw = reinterpret_cast<double*>(taskData->inputs[0]);

  for (int i = 0; i < input_size_; ++i) {
    for (int j = 0; j < input_size_; ++j) {
      A_[i][j] = A_raw[i * input_size_ + j];
    }
  }

  auto* B_raw = reinterpret_cast<double*>(taskData->inputs[1]);
  b_.resize(input_size_);
  for (int i = 0; i < input_size_; ++i) {
    b_[i] = B_raw[i];
  }

  if (!isDiagonallyDominant()) return false;
  if (!transformSystem()) return false;
  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTask::run() {
  internal_order_test();

  std::vector<double> x_new(input_size_, 0.0);
  double max_diff = 0.0;

  do {
    max_diff = 0.0;

    for (int i = 0; i < input_size_; ++i) {
      double sum = 0.0;

      for (int j = 0; j < input_size_; ++j) {
        if (i != j) {
          sum += B_[i][j] * x_[j];
        }
      }

      x_new[i] = g_[i] + sum;

      double diff = std::abs(x_new[i] - x_[i]);
      max_diff = std::max(max_diff, diff);
    }

    x_ = x_new;

  } while (max_diff > tolerance_);

  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTask::post_processing() {
  internal_order_test();
  for (int i = 0; i < static_cast<int>(x_.size()); i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x_[i];
  }
  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI::isDiagonallyDominant() {
  if (world.rank() == 0) {
    for (int i = 0; i < input_size_; ++i) {
      double diagonal = std::abs(A_[i][i]);
      double sum = 0.0;

      for (int j = 0; j < input_size_; ++j) {
        if (i != j) {
          sum += std::abs(A_[i][j]);
        }
      }

      if (diagonal <= sum) {
        return false;
      }
    }
  }
  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI::transformSystem() {
  if (world.rank() == 0) {
    B_.resize(input_size_ * input_size_);
    g_.resize(input_size_);

    for (int i = 0; i < input_size_; ++i) {
      double diagonal = A_[i][i];
      if (diagonal == 0.0) {
        std::cerr << "Zero diagonal element detected, cannot apply iteration." << std::endl;
        return false;
      }

      g_[i] = b_[i] / diagonal;
      for (int j = 0; j < input_size_; ++j) {
        if (i != j) {
          B_[i * input_size_ + j] = -A_[i][j] / diagonal;
        }
      }
    }
  }
  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* X_raw = reinterpret_cast<double*>(taskData->outputs[0]);
    x_.resize(input_size_);
    for (int i = 0; i < input_size_; ++i) {
      x_[i] = X_raw[i];
    }
  }
  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] != taskData->inputs_count[1] ||
        taskData->inputs_count[0] != taskData->outputs_count[0]) {
      return false;
    }
    input_size_ = taskData->inputs_count[0];
    A_.resize(input_size_, std::vector<double>(input_size_));
    b_.resize(input_size_);
    auto* A_raw = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* B_raw = reinterpret_cast<double*>(taskData->inputs[1]);
    for (int i = 0; i < input_size_; ++i) {
      for (int j = 0; j < input_size_; ++j) {
        A_[i][j] = A_raw[i * input_size_ + j];
      }
      b_[i] = B_raw[i];
    }
    if (!isDiagonallyDominant()) return false;
    if (!transformSystem()) return false;
  }
  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI::run() {
  internal_order_test();

  boost::mpi::broadcast(world, input_size_, 0);
  x_.resize(input_size_);

  std::vector<int> local_matrix_elements(world.size());
  std::vector<int> offsets_matrix(world.size());
  std::vector<int> right_side_values(world.size());
  std::vector<int> offsets_right_side(world.size());

  int main = input_size_ / world.size();
  int extra = input_size_ % world.size();
  int offset = 0;
  for (int proc = 0; proc < world.size(); ++proc) {
    local_matrix_elements[proc] = right_side_values[proc] = (proc < extra ? main + 1 : main);
    offsets_matrix[proc] = offsets_right_side[proc] = offset;
    offset += local_matrix_elements[proc];
  }
  for (int i = 0; i < static_cast<int>(local_matrix_elements.size()); i++) {
    local_matrix_elements[i] *= input_size_;
    offsets_matrix[i] *= input_size_;
  }

  std::vector<double> x_new(input_size_, 0.0);
  std::vector<double> local_current(right_side_values[world.rank()], 0.0);

  std::vector<double> B_local(local_matrix_elements[world.rank()], 0.0);
  std::vector<double> g_local(right_side_values[world.rank()], 0.0);

  boost::mpi::scatterv(world, B_.data(), local_matrix_elements, offsets_matrix, B_local.data(),
                       local_matrix_elements[world.rank()] * input_size_, 0);
  boost::mpi::scatterv(world, g_.data(), right_side_values, offsets_right_side, g_local.data(),
                       right_side_values[world.rank()], 0);
  double max_diff;
  do {
    max_diff = 0.0;
    boost::mpi::broadcast(world, x_new.data(), x_new.size(), 0);
    for (int iter_place = 0; iter_place < right_side_values[world.rank()]; ++iter_place) {
      double iter_sum = 0.0;
      for (int j = 0; j < input_size_; ++j) {
        if (j != (offsets_right_side[world.rank()] + iter_place)) {
          iter_sum += B_local[iter_place * input_size_ + j] * x_new[j];
        }
      }
      local_current[iter_place] = g_local[iter_place] + iter_sum;
    }

    boost::mpi::gatherv(world, local_current.data(), local_current.size(),
                        x_new.data() + offsets_right_side[world.rank()], right_side_values, 0);

    if (world.rank() == 0) {
      for (size_t k = 0; k < x_new.size(); ++k) {
        double diff = std::abs(x_new[k] - x_[k]);
        max_diff = std::max(max_diff, diff);
      }
      x_ = x_new;
    }
    boost::mpi::broadcast(world, max_diff, 0);
  } while (max_diff > tolerance_);

  return true;
}

bool lysov_i_simple_iteration_method_mpi::SlaeIterationTaskMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    for (int i = 0; i < static_cast<int>(x_.size()); ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = x_[i];
    }
  }
  return true;
}