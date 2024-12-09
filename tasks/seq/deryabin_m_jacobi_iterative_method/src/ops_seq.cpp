#include "seq/deryabin_m_jacobi_iterative_method/include/ops_seq.hpp"

#include <thread>

bool deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential::pre_processing() {
  internal_order_test();
  input_matrix_ = reinterpret_cast<std::vector<double> *>(taskData->inputs[0])[0];
  input_right_vector_ = reinterpret_cast<std::vector<double> *>(taskData->inputs[1])[0];
  output_x_vector_ = std::vector<double>(input_right_vector_.size());
  return true;
}

bool deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential::validation() {
  internal_order_test();
  std::vector<double> matrix_ = reinterpret_cast<std::vector<double> *>(taskData->inputs[0])[0];
  unsigned short i = 0;
  auto lambda = [&](double first, double second) { return (std::abs(first) + std::abs(second)); };
  while (i != sqrt(matrix_.size())) {
    if (i == 0) {
      if (std::abs(matrix_[0]) <=
          std::accumulate(matrix_.begin() + 1, matrix_.begin() + sqrt(matrix_.size()) - 1, 0, lambda)) {
        return false;
      }
    }
    if (i > 0 && i < sqrt(matrix_.size()) - 1) {
      if (std::abs(matrix_[i * (sqrt(matrix_.size()) + 1)]) <=
          std::accumulate(matrix_.begin() + i * sqrt(matrix_.size()),
                          matrix_.begin() + i * (sqrt(matrix_.size()) + 1) - 1, 0, lambda) +
              std::accumulate(matrix_.begin() + i * (sqrt(matrix_.size()) + 1) + 1,
                              matrix_.begin() + (i + 1) * sqrt(matrix_.size()) - 1, 0, lambda)) {
        return false;
      }
    }
    if (i == sqrt(matrix_.size()) - 1) {
      if (std::abs(matrix_[i * (sqrt(matrix_.size()) + 1)]) <=
          std::accumulate(matrix_.begin() + i * sqrt(matrix_.size()), matrix_.end() - 1, 0, lambda)) {
        return false;
      }
    }
    i++;
  }
  return taskData->inputs_count[0] == 1 && taskData->outputs_count[0] == 1 && taskData->inputs_count[1] == 1;
}

bool deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential::run() {
  internal_order_test();
  unsigned short Nmax = 10000, num_of_iterations = 0;
  double epsilon = pow(10, -6), max_delta_x_i = 0;
  std::vector<double> x_old;
  do {
    x_old = output_x_vector_;
    unsigned short i = 0, j;
    double sum;
    while (i != sqrt(input_matrix_.size())) {
      j = 0;
      sum = 0;
      while (j != sqrt(input_matrix_.size())) {
        if (i != j) {
          sum += input_matrix_[i * sqrt(input_matrix_.size()) + j] * x_old[j];
        }
        j++;
      }
      output_x_vector_[i] =
          (input_right_vector_[i] - sum) * (1.0 / input_matrix_[i * (sqrt(input_matrix_.size()) + 1)]);
      if (std::abs(output_x_vector_[i] - x_old[i]) > max_delta_x_i) {
        max_delta_x_i = std::abs(output_x_vector_[i] - x_old[i]);
      }
      i++;
    }
    num_of_iterations++;
  } while (num_of_iterations < Nmax && max_delta_x_i > epsilon);
  return true;
}

bool deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<double> *>(taskData->outputs[0])[0] = output_x_vector_;
  return true;
}
