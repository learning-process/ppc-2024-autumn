// Copyright 2024 Sdobnov Vladimir
#include "mpi/Sdobnov_V_iteration_method_yakoby/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

std::vector<double> Sdobnov_iteration_method_yakoby_par::iteration_method_yakoby(int n, const std::vector<double>& A,
                                                                                 const std::vector<double>& b,
                                                                                 double tolerance, int maxIterations) {
  std::vector<double> x(n, 0.0);
  std::vector<double> x_new(n, 0.0);

  for (int iter = 0; iter < maxIterations; ++iter) {
    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          sum += A[i * n + j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[i * n + i];
    }
    double max_diff = 0.0;

    for (int i = 0; i < n; ++i) {
      max_diff = std::max(max_diff, fabs(x_new[i] - x[i]));
    }

    if (max_diff < tolerance) {
      break;
    }

    x = x_new;
  }

  return x_new;
}

static void calculate_mat_part_param(int rows, int num_proc, std::vector<int>& sizes, std::vector<int>& offsets) {
  sizes.resize(num_proc, 0);
  offsets.resize(num_proc, -1);
  if (num_proc > rows) {
    for (int i = 0; i < rows; ++i) {
      sizes[i] = rows;
      offsets[i] = i * rows;
    }
  } else {
    int a = rows / num_proc;
    int b = rows % num_proc;
    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1) * rows;
      } else {
        sizes[i] = a * rows;
      }
      offsets[i] = offset;
      offset += sizes[i];
    }
  }
}

static void calculate_free_members_part_param(int len, int num_proc, std::vector<int>& sizes,
                                              std::vector<int>& offsets) {
  sizes.resize(num_proc, 0);
  offsets.resize(num_proc, -1);
  if (num_proc > len) {
    for (int i = 0; i < len; ++i) {
      sizes[i] = 1;
      offsets[i] = i;
    }
  } else {
    int a = len / num_proc;
    int b = len % num_proc;
    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (b-- > 0) {
        sizes[i] = (a + 1);
      } else {
        sizes[i] = a;
      }
      offsets[i] = offset;
      offset += sizes[i];
    }
  }
}

static bool isDiagonallyDominant(int n, const std::vector<double>& A) {
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;

    for (int j = 0; j < n; ++j) {
      if (i != j) {
        sum += fabs(A[i * n + j]);
      }
    }

    if (fabs(A[i * n + i]) <= sum) {
      return false;
    }

    if (fabs(A[i * n + i]) < 1e-9) {
      return false;
    }
  }

  return true;
}

bool Sdobnov_iteration_method_yakoby_par::IterationMethodYakobySeq::pre_processing() {
  internal_order_test();

  size_ = taskData->inputs_count[0];
  matrix_.assign(size_ * size_, 0.0);
  free_members_.assign(size_, 0.0);
  res_.assign(size_, 0.0);

  auto* pmatrix = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* pfree_members = reinterpret_cast<double*>(taskData->inputs[1]);

  std::copy(pmatrix, pmatrix + size_ * size_, matrix_.begin());
  std::copy(pfree_members, pfree_members + size_, free_members_.begin());

  return true;
}

bool Sdobnov_iteration_method_yakoby_par::IterationMethodYakobySeq::validation() {
  internal_order_test();

  bool correct_count = taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0 &&
                       taskData->inputs.size() == 2 && taskData->outputs_count.size() == 1 &&
                       taskData->outputs_count[0] >= 0 && taskData->outputs.size() == 1;
  if (!correct_count) return false;

  int size = taskData->inputs_count[0];
  std::vector<double> matrix(size * size, 0.0);
  auto* pmatrix = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(pmatrix, pmatrix + size * size, matrix.begin());

  return isDiagonallyDominant(size, matrix);
}

bool Sdobnov_iteration_method_yakoby_par::IterationMethodYakobySeq::run() {
  internal_order_test();
  res_ = iteration_method_yakoby(size_, matrix_, free_members_, tolerance, maxIterations);
  return true;
}

bool Sdobnov_iteration_method_yakoby_par::IterationMethodYakobySeq::post_processing() {
  internal_order_test();
  for (int i = 0; i < size_; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = res_[i];
  }
  return true;
}

bool Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar::pre_processing() {
  internal_order_test();

  mat_part_sizes.resize(world.size());
  mat_part_offsets.resize(world.size());
  free_members_part_sizes.resize(world.size());
  free_members_part_offsets.resize(world.size());

  if (world.rank() == 0) {
    size_ = taskData->inputs_count[0];
    matrix_.assign(size_ * size_, 0.0);
    free_members_.assign(size_, 0.0);
    res_.assign(size_, 0.0);
    last_res.assign(size_, 0.0);

    auto* pmatrix = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* pfree_members = reinterpret_cast<double*>(taskData->inputs[1]);

    std::copy(pmatrix, pmatrix + size_ * size_, matrix_.begin());
    std::copy(pfree_members, pfree_members + size_, free_members_.begin());

    calculate_mat_part_param(size_, world.size(), mat_part_sizes, mat_part_offsets);
    calculate_free_members_part_param(size_, world.size(), free_members_part_sizes, free_members_part_offsets);
  }

  return true;
}

bool Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    bool correct_count = taskData->inputs_count.size() == 1 && taskData->inputs_count[0] >= 0 &&
                         taskData->inputs.size() == 2 && taskData->outputs_count.size() == 1 &&
                         taskData->outputs_count[0] >= 0 && taskData->outputs.size() == 1;
    if (!correct_count) return false;

    int size = taskData->inputs_count[0];
    std::vector<double> matrix(size * size, 0.0);
    auto* pmatrix = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(pmatrix, pmatrix + size * size, matrix.begin());
    return isDiagonallyDominant(size, matrix);
  }

  return true;
}

bool Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar::run() {
  internal_order_test();

  boost::mpi::broadcast(world, mat_part_sizes, 0);
  boost::mpi::broadcast(world, mat_part_offsets, 0);
  boost::mpi::broadcast(world, free_members_part_sizes, 0);
  boost::mpi::broadcast(world, free_members_part_offsets, 0);
  boost::mpi::broadcast(world, size_, 0);

  int l_mat_part_size = mat_part_sizes[world.rank()];
  int l_free_members_part_size = free_members_part_sizes[world.rank()];

  l_matrix.resize(l_mat_part_size);
  l_free_members.resize(l_free_members_part_size);
  l_res.resize(l_free_members_part_size);

  boost::mpi::scatterv(world, matrix_.data(), mat_part_sizes, mat_part_offsets, l_matrix.data(), l_mat_part_size, 0);
  boost::mpi::scatterv(world, free_members_.data(), free_members_part_sizes, free_members_part_offsets,
                       l_free_members.data(), l_free_members_part_size, 0);

  for (int iter = 0; iter < maxIterations; iter++) {
    if (world.rank() == 0) {
      std::copy(res_.begin(), res_.end(), last_res.begin());
    }
    boost::mpi::broadcast(world, last_res, 0);

    for (int i = 0; i < free_members_part_sizes[world.rank()]; i++) {
      double sum = 0;
      for (int j = 0; j < size_; j++) {
        if (j != (free_members_part_offsets[world.rank()] + i)) {
          sum += l_matrix[i * size_ + j] * last_res[j];
        }
      }
      l_res[i] = (l_free_members[i] - sum) / l_matrix[i * size_ + free_members_part_offsets[world.rank()] + i];
    }

    boost::mpi::gatherv(world, l_res.data(), free_members_part_sizes[world.rank()], res_.data(),
                        free_members_part_sizes, free_members_part_offsets, 0);

    bool stop_flag = false;

    if (world.rank() == 0) {
      double max_diff = 0.0;
      for (int i = 0; i < size_; ++i) {
        max_diff = std::max(max_diff, fabs(res_[i] - last_res[i]));
      }

      if (max_diff < tolerance) {
        stop_flag = true;
      }
    }
    boost::mpi::broadcast(world, stop_flag, 0);
    if (stop_flag) break;
  }

  return true;
}

bool Sdobnov_iteration_method_yakoby_par::IterationMethodYakobyPar::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < size_; ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = res_[i];
    }
  }
  return true;
}