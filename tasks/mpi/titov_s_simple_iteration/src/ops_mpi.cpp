// Copyright 2023 Nesterov Alexander
#include "mpi/titov_s_simple_iteration/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

using namespace std::chrono_literals;

void titov_s_simple_iteration_mpi::MPISimpleIterationSequential::transformMatrix() {
  for (unsigned int i = 0; i < rows_; ++i) {
    float diagonal = input_[i * cols_ + i];
    if (diagonal == 0.0f) {
      throw std::runtime_error("Diagonal element is zero, cannot transform matrix.");
    }
    for (unsigned int j = 0; j < cols_ - 1; ++j) {
      if (i != j) {
        input_[i * cols_ + j] = -input_[i * cols_ + j] / diagonal;
      } else {
        input_[i * cols_ + j] = 0.0f;
      }
    }
    input_[i * cols_ + cols_ - 1] /= diagonal;
  }
}

bool titov_s_simple_iteration_mpi::MPISimpleIterationSequential::isDiagonallyDominant() {
  for (unsigned int i = 0; i < rows_; ++i) {
    float diagonal = std::abs(input_[i * cols_ + i]);
    float sum = 0.0f;
    for (unsigned int j = 0; j < cols_ - 1; ++j) {
      if (i != j) {
        sum += std::abs(input_[i * cols_ + j]);
      }
    }
    if (diagonal <= sum) {
      return false;
    }
  }
  return true;
}

bool titov_s_simple_iteration_mpi::MPISimpleIterationSequential::pre_processing() {
  internal_order_test();

  rows_ = taskData->inputs_count[0];
  cols_ = taskData->inputs_count[1];

  input_ = std::make_unique<float[]>(rows_ * cols_);
  res_ = std::make_unique<float[]>(rows_);

  for (unsigned int i = 0; i < rows_; i++) {
    auto* tmp_ptr = reinterpret_cast<float*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < cols_; j++) {
      input_[i * cols_ + j] = tmp_ptr[j];
    }
  }

  auto* epsilon_ptr = reinterpret_cast<float*>(taskData->inputs[rows_]);
  epsilon_ = *epsilon_ptr;
  if (!isDiagonallyDominant()) {
    std::cerr << "Matrix is not diagonally dominant. The method may not converge.\n";
    return false;
  }

  transformMatrix();

  return true;
}

bool titov_s_simple_iteration_mpi::MPISimpleIterationSequential::validation() {
  internal_order_test();

  if (taskData->inputs_count.empty() || taskData->inputs.empty()) {
    std::cerr << "Validation failed: inputs_count or inputs is empty.\n";
    return false;
  }

  unsigned int rows = taskData->inputs_count[0];
  unsigned int cols = taskData->inputs_count[1];

  if (taskData->inputs.size() < rows + 1) {
    return false;
  }

  if (rows == 0 || cols == 0) {
    return false;
  }

  for (unsigned int i = 0; i < rows; ++i) {
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[i]);
    if (tmp_ptr == nullptr) {
      return false;
    }
  }

  auto* epsilon_ptr = reinterpret_cast<float*>(taskData->inputs[rows]);
  if (epsilon_ptr == nullptr) {
    return false;
  }
  float epsilon = *epsilon_ptr;
  if (epsilon <= 0.0f || epsilon > 1.0f) {
    return false;
  }

  if (taskData->outputs_count.empty() || taskData->outputs_count[0] < 1) {
    return false;
  }

  return true;
}

bool titov_s_simple_iteration_mpi::MPISimpleIterationSequential::run() {
  internal_order_test();

  std::unique_ptr<float[]> x_prev = std::make_unique<float[]>(rows_);
  std::unique_ptr<float[]> x_curr = std::make_unique<float[]>(rows_);
  std::fill(x_prev.get(), x_prev.get() + rows_, 0.0f);

  float max_diff;

  do {
    max_diff = 0.0f;

    for (unsigned int i = 0; i < rows_; i++) {
      float new_x = input_[i * cols_ + cols_ - 1];

      for (unsigned int j = 0; j < cols_ - 1; j++) {
        new_x += input_[i * cols_ + j] * x_prev[j];
      }

      x_curr[i] = new_x;

      float diff = std::abs(x_curr[i] - x_prev[i]);
      max_diff = std::max(max_diff, diff);
    }

    std::copy(x_curr.get(), x_curr.get() + rows_, x_prev.get());

  } while (max_diff > epsilon_);

  std::copy(x_curr.get(), x_curr.get() + rows_, res_.get());

  return true;
}

bool titov_s_simple_iteration_mpi::MPISimpleIterationSequential::post_processing() {
  internal_order_test();

  auto* output_ptr = reinterpret_cast<float*>(taskData->outputs[0]);

  for (unsigned int i = 0; i < rows_; ++i) {
    output_ptr[i] = res_[i];
  }

  return true;
}

bool titov_s_simple_iteration_mpi::MPISimpleIterationParallel::pre_processing() {
  internal_order_test();

  number_matrix.resize(world.size()), offset_matrix.resize(world.size()), number_values.resize(world.size()),
      offset_values.resize(world.size());

  if (world.rank() == 0) {
    Rows = *reinterpret_cast<size_t*>(taskData->inputs[2]);
    epsilon_ = *reinterpret_cast<double*>(taskData->inputs[3]);

    auto* Matrix_input = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* Values_input = reinterpret_cast<double*>(taskData->inputs[1]);

    Matrix.assign(Matrix_input, Matrix_input + Rows * Rows);
    Values.assign(Values_input, Values_input + Rows);
    current.assign(Rows, 0.0);
    prev.assign(Rows, 0.0);
    int bias = 0;
    int main = Rows / world.size();
    int extra = Rows % world.size();
    for (int proc = 0; proc < world.size(); ++proc) {
      int proc_rows = main + (extra-- > 0 ? 1 : 0);
      number_matrix[proc] = proc_rows * Rows;
      offset_matrix[proc] = bias;
      bias += number_matrix[proc];
    }

    main = Rows / world.size();
    extra = Rows % world.size();
    bias = 0;

    for (int proc = 0; proc < world.size(); ++proc) {
      number_values[proc] = main + (extra-- > 0 ? 1 : 0);
      offset_values[proc] = bias;
      bias += number_values[proc];
    }
  }
  return true;
}

bool titov_s_simple_iteration_mpi::MPISimpleIterationParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    Rows = *reinterpret_cast<size_t*>(taskData->inputs[2]);
    if (Rows == 0) {
      return false;
    }
    if (taskData->inputs_count.size() != 4 || taskData->outputs_count.size() != 1) {
      return false;
    }
    epsilon_ = *reinterpret_cast<double*>(taskData->inputs[3]);
    if (epsilon_ >= 1) {
      return false;
    }
    auto* Matrixinput = reinterpret_cast<double*>(taskData->inputs[0]);
    Matrix.assign(Matrixinput, Matrixinput + Rows * Rows);
    if (!isDiagonallyDominant()) {
      return false;
    }
  }
  return true;
}

bool titov_s_simple_iteration_mpi::MPISimpleIterationParallel::run() {
  internal_order_test();
  std::vector<double> current_l;

  boost::mpi::broadcast(world, number_matrix, 0);
  boost::mpi::broadcast(world, number_values, 0);
  boost::mpi::broadcast(world, offset_values, 0);
  boost::mpi::broadcast(world, Rows, 0);
  int Matrix_size_l = number_matrix[world.rank()];
  int Values_size_l = number_values[world.rank()];
  Matrix_l.resize(Matrix_size_l);
  Values_l.resize(Values_size_l);
  current_l.resize(number_values[world.rank()]);
  bool end;
  if (world.rank() == 0) {
    boost::mpi::scatterv(world, Matrix.data(), number_matrix, offset_matrix, Matrix_l.data(), Matrix_size_l, 0);
    boost::mpi::scatterv(world, Values.data(), number_values, offset_values, Values_l.data(), Values_size_l, 0);
  } else {
    boost::mpi::scatterv(world, Matrix_l.data(), Matrix_size_l, 0);
    boost::mpi::scatterv(world, Values_l.data(), Values_size_l, 0);
  }

  end = false;
  do {
    if (world.rank() == 0) {
      std::copy(current.begin(), current.end(), prev.begin());
    }
    boost::mpi::broadcast(world, prev, 0);
    double iter;
    for (int iter_place = 0; iter_place < number_values[world.rank()]; iter_place++) {
      iter = 0;
      for (int j = 0; j < Rows; j++) {
        if (j != (offset_values[world.rank()] + iter_place)) {
          iter += Matrix_l[iter_place * Rows + j] * prev[j];
        }
      }

      int global_row = offset_values[world.rank()] + iter_place;

      double iter_sum = Values_l[iter_place] - iter;

      double diagonal_element = Matrix_l[iter_place * Rows + global_row];
      current_l[iter_place] = iter_sum / diagonal_element;
    }

    if (world.rank() == 0) {
      boost::mpi::gatherv(world, current_l.data(), number_values[world.rank()], current.data(), number_values,
                          offset_values, 0);
    } else {
      boost::mpi::gatherv(world, current_l.data(), number_values[world.rank()], 0);
    }

    if (world.rank() == 0) {
      double max_diff = 0.0;

      for (size_t k = 0; k < prev.size(); k++) {
        double diff = std::abs(current[k] - prev[k]);
        if (diff > max_diff) {
          max_diff = diff;
        }
      }
      end = (max_diff < epsilon_);
    }
    boost::mpi::broadcast(world, end, 0);
  } while (!end);

  return true;
}

bool titov_s_simple_iteration_mpi::MPISimpleIterationParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < current.size(); ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = current[i];
    }
  }
  return true;
}

bool titov_s_simple_iteration_mpi::MPISimpleIterationParallel::isDiagonallyDominant() {
  double X;
  double S;

  for (int i = 0; i < Rows; ++i) {
    X = std::abs(Matrix[i * Rows + i]);
    S = 0.0;

    for (int j = 0; j < Rows; ++j) {
      if (i != j) {
        S += std::abs(Matrix[i * Rows + j]);
      }
    }
    if (X <= S) {
      return false;
    }
  }
  return true;
}
