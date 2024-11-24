// Copyright 2023 Nesterov Alexander
#include "mpi/muhina_m_horizontal_cheme/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

using namespace std::chrono_literals;

std::vector<int> muhina_m_horizontal_cheme_mpi::matrixVectorMultiplication(const std::vector<int>& matrix,
                                                                           const std::vector<int>& vec, int rows,
                                                                           int cols) {
  std::vector<int> result(rows, 0);

  for (int i = 0; i < rows; ++i) {
    int row_result = 0;
    for (int j = 0; j < cols; ++j) {
      row_result += matrix[i * cols + j] * vec[j];
    }
    result[i] = row_result;
  }

  return result;
}

void muhina_m_horizontal_cheme_mpi::calculate_distribution(int rows, int cols, int num_proc, std::vector<int>& sizes,
                                                           std::vector<int>& displs) {
  sizes.resize(num_proc, 0);
  displs.resize(num_proc, -1);

  if (num_proc > rows) {
    for (int i = 0; i < rows; ++i) {
      sizes[i] = cols;
      displs[i] = i * cols;
    }
  } else {
    int rows_per_proc = rows / num_proc;
    int extra_rows = rows % num_proc;

    int offset = 0;
    for (int i = 0; i < num_proc; ++i) {
      if (extra_rows > 0) {
        sizes[i] = (rows_per_proc + 1) * cols;
        --extra_rows;
      } else {
        sizes[i] = rows_per_proc * cols;
      }
      displs[i] = offset;
      offset += sizes[i];
    }
  }
}

bool muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPISequential::pre_processing() {
  internal_order_test();

  int* m_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int m_size = taskData->inputs_count[0];

  int* v_data = reinterpret_cast<int*>(taskData->inputs[1]);
  int v_size = taskData->inputs_count[1];

  matrix_.assign(m_data, m_data + m_size);
  vec_.assign(v_data, v_data + v_size);

  return true;
}

bool muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPISequential::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] == 0 || taskData->inputs_count[1] == 0) {
    return false;
  }
  if (taskData->inputs_count[0] % taskData->inputs_count[1] != 0) {
    return false;
  }
  if (taskData->inputs_count[0] / taskData->inputs_count[1] != taskData->outputs_count[0]) {
    return false;
  }
  return true;
}

bool muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPISequential::run() {
  internal_order_test();
  int cols = taskData->inputs_count[1];
  int rows = taskData->inputs_count[0] / taskData->inputs_count[1];

  result_ = matrixVectorMultiplication(matrix_, vec_, rows, cols);
  return true;
}

bool muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPISequential::post_processing() {
  internal_order_test();
  int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_.begin(), result_.end(), output_data);
  return true;
}

bool muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPIParallel::pre_processing() {
  internal_order_test();

  if (world_.rank() == 0) {
    int* m_data = reinterpret_cast<int*>(taskData->inputs[0]);
    int m_size = taskData->inputs_count[0];
    int* v_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int v_size = taskData->inputs_count[1];

    rows_ = v_size;
    cols_ = m_size / rows_;

    matrix_.assign(m_data, m_data + m_size);
    vec_.assign(v_data, v_data + v_size);
    result_.resize(cols_, 0);

    calculate_distribution(cols_, rows_, world_.size(), distribution, displacement);
  }

  boost::mpi::broadcast(world_, cols_, 0);
  boost::mpi::broadcast(world_, rows_, 0);
  boost::mpi::broadcast(world_, distribution, 0);
  boost::mpi::broadcast(world_, displacement, 0);

  if (world_.rank() != 0) {
    matrix_.resize(cols_ * rows_);
    vec_.resize(rows_);
    result_.resize(cols_);
  }

  return true;
}

bool muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPIParallel::validation() {
  internal_order_test();
  if (world_.rank() == 0) {
    if (taskData->inputs_count[0] == 0 || taskData->inputs_count[1] == 0) {
      return false;
    }
    if (taskData->inputs_count[0] % taskData->inputs_count[1] != 0) {
      return false;
    }
    if (taskData->inputs_count[0] / taskData->inputs_count[1] != taskData->outputs_count[0]) {
      return false;
    }
  }
  return true;
}

bool muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPIParallel::run() {
  internal_order_test();
  boost::mpi::broadcast(world_, matrix_, 0);
  boost::mpi::broadcast(world_, vec_, 0);

  int local_start_row = displacement[world_.rank()] / cols_;
  int local_rows = distribution[world_.rank()] / cols_;
  std::vector<int> local_result(cols_, 0);

  for (int i = 0; i < cols_; ++i) {
    for (int j = 0; j < local_rows; ++j) {
      int row = local_start_row + j;
      local_result[i] += matrix_[i * rows_ + row] * vec_[row];
    }
  }

  if (world_.rank() == 0) {
    result_.assign(cols_, 0);
  }

  boost::mpi::reduce(world_, local_result.data(), cols_, result_.data(), std::plus<>(), 0);

  return true;
}

bool muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPIParallel::post_processing() {
  internal_order_test();

  if (world_.rank() == 0) {
    int* output_data = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result_.begin(), result_.end(), output_data);
  }

  return true;
}
