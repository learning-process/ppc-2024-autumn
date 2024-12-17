// Copyright 2023 Nesterov Alexander
#include "mpi/malyshev_v_conjugate_gradient_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"

bool malyshev_v_conjugate_gradient_method::MPIConjugateGradientSequential::pre_processing() {
  internal_order_test();

  rows_ = taskData->inputs_count[0];
  cols_ = taskData->inputs_count[1];

  input_ = std::make_unique<double[]>(rows_ * cols_);
  res_ = std::make_unique<double[]>(rows_);

  for (unsigned int i = 0; i < rows_; i++) {
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < cols_; j++) {
      input_[i * cols_ + j] = tmp_ptr[j];
    }
  }

  auto* values_ptr = reinterpret_cast<double*>(taskData->inputs[rows_]);
  values_.assign(values_ptr, values_ptr + rows_);

  auto* epsilon_ptr = reinterpret_cast<double*>(taskData->inputs[rows_ + 1]);
  epsilon_ = *epsilon_ptr;
  return true;
}

bool malyshev_v_conjugate_gradient_method::MPIConjugateGradientSequential::validation() {
  internal_order_test();

  rows_ = taskData->inputs_count[0];
  cols_ = taskData->inputs_count[1];

  input_ = std::make_unique<double[]>(rows_ * cols_);
  res_ = std::make_unique<double[]>(rows_);

  for (unsigned int i = 0; i < rows_; i++) {
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[i]);
    for (unsigned int j = 0; j < cols_; j++) {
      input_[i * cols_ + j] = tmp_ptr[j];
    }
  }

  auto* values_ptr = reinterpret_cast<double*>(taskData->inputs[rows_]);
  values_.assign(values_ptr, values_ptr + rows_);

  auto* epsilon_ptr = reinterpret_cast<double*>(taskData->inputs[rows_ + 1]);
  epsilon_ = *epsilon_ptr;

  return true;
}

bool malyshev_v_conjugate_gradient_method::MPIConjugateGradientSequential::run() {
  internal_order_test();

  std::vector<double> x(rows_, 0.0);
  std::vector<double> r = values_;
  std::vector<double> p = r;
  double rs_old = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

  for (unsigned int k = 0; k < rows_; ++k) {
    std::vector<double> Ap(rows_, 0.0);
    for (unsigned int i = 0; i < rows_; ++i) {
      for (unsigned int j = 0; j < rows_; ++j) {
        Ap[i] += input_[i * rows_ + j] * p[j];
      }
    }

    double alpha = rs_old / std::inner_product(p.begin(), p.end(), Ap.begin(), 0.0);

    for (unsigned int i = 0; i < rows_; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double rs_new = std::inner_product(r.begin(), r.end(), r.begin(), 0.0);

    if (std::sqrt(rs_new) < epsilon_) {
      break;
    }

    for (unsigned int i = 0; i < rows_; ++i) {
      p[i] = r[i] + (rs_new / rs_old) * p[i];
    }

    rs_old = rs_new;
  }

  std::copy(x.begin(), x.end(), res_.get());
  return true;
}

bool malyshev_v_conjugate_gradient_method::MPIConjugateGradientSequential::post_processing() {
  internal_order_test();

  auto* output_ptr = reinterpret_cast<double*>(taskData->outputs[0]);

  for (unsigned int i = 0; i < rows_; ++i) {
    output_ptr[i] = res_[i];
  }

  return true;
}

bool malyshev_v_conjugate_gradient_method::MPIConjugateGradientParallel::pre_processing() {
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

bool malyshev_v_conjugate_gradient_method::MPIConjugateGradientParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs_count.empty() || taskData->inputs.empty()) {
      return false;
    }
    Rows = *reinterpret_cast<size_t*>(taskData->inputs[2]);
    if (taskData->inputs_count.size() != 4 || taskData->outputs_count.size() != 1) {
      return false;
    }
    epsilon_ = *reinterpret_cast<double*>(taskData->inputs[3]);
    if (epsilon_ >= 1) {
      return false;
    }
    auto* Matrixinput = reinterpret_cast<double*>(taskData->inputs[0]);
    Matrix.assign(Matrixinput, Matrixinput + Rows * Rows);
  }
  return true;
}

bool malyshev_v_conjugate_gradient_method::MPIConjugateGradientParallel::run() {
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

bool malyshev_v_conjugate_gradient_method::MPIConjugateGradientParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < current.size(); ++i) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = current[i];
    }
  }
  return true;
}