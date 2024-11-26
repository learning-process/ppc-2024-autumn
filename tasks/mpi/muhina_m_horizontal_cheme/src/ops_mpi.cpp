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
    for (int i = 0; i < num_proc; ++i) {
      if (i < rows) {
        sizes[i] = cols;
        displs[i] = i * cols;
      } else {
        sizes[i] = 0;
        displs[i] = -1;
      }
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

  matrix_.clear();
  vec_.clear();

  if (world_.rank() == 0) {
    int* m_data = reinterpret_cast<int*>(taskData->inputs[0]);
    int m_size = taskData->inputs_count[0];
    int* v_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int v_size = taskData->inputs_count[1];

    rows_ = v_size;
    cols_ = m_size / rows_;

    matrix_.insert(matrix_.end(), m_data, m_data + m_size);
    vec_.insert(vec_.end(), v_data, v_data + v_size);
  }
  result_.clear();

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

  boost::mpi::broadcast(world_, cols_, 0);
  boost::mpi::broadcast(world_, rows_, 0);

  int delta;
  int ost;
  if (world_.size() == 1) {
    delta = 0;
    ost = rows_;
  } else {
    delta = rows_ / (world_.size() - 1);
    ost = rows_ % (world_.size() - 1);
  }
  std::vector<int> localMatrix;

  if (world_.rank() != 0) {
    localMatrix.resize(delta * cols_, 0);
    vec_.resize(rows_);
  }

  boost::mpi::broadcast(world_, vec_.data(), vec_.size(), 0);

  if (world_.rank() == 0) {
    for (int proc = 0; proc < world_.size() - 1; proc++) {
      world_.send(proc + 1, 0, matrix_.data() + proc * delta * cols_ + ost * cols_, delta * cols_);
    }
    localMatrix.insert(localMatrix.end(), matrix_.data(), matrix_.data() + cols_ * ost);
  } else {
    world_.recv(0, 0, localMatrix.data(), delta * cols_);
  }

  std::vector<int> local_result(localMatrix.size() / cols_, 0);

  for (int i = 0; i < (int)(localMatrix.size() / cols_); i++) {
    for (int j = 0; j < cols_; j++) {
      local_result[i] += localMatrix[i * cols_ + j] * vec_[j];
    }
  }

  if (world_.rank() == 0) {
    result_.clear();
    result_.insert(result_.end(), local_result.data(), local_result.data() + ost);
    for (int proc = 1; proc < world_.size(); proc++) {
      std::vector<int> temp(delta);
      world_.recv(proc, 0, temp.data(), delta);
      result_.insert(result_.end(), temp.data(), temp.data() + delta);
    }
  } else {
    world_.send(0, 0, local_result.data(), delta);
  }

  return true;
}

bool muhina_m_horizontal_cheme_mpi::HorizontalSchemeMPIParallel::post_processing() {
  internal_order_test();

  if (world_.rank() == 0) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_.data()));
  }

  return true;
}