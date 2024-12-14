#include "seq/deryabin_m_jacobi_iterative_method/include/ops_seq.hpp"

#include <thread>

bool deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential::pre_processing() {
  internal_order_test();
  input_right_vector_ = reinterpret_cast<std::vector<double> *>(taskData->inputs[1])[0];
  output_x_vector_ = std::vector<double>(input_right_vector_.size());
  return true;
}

bool deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential::validation() {
  internal_order_test();
  input_matrix_ = reinterpret_cast<std::vector<double> *>(taskData->inputs[0])[0];
  unsigned short i = 0;
  unsigned short j;
  auto n = (unsigned short)(sqrt(input_matrix_.size()));
  while (i != n) {
    if (i == 0) {
      j = 0;
      while (j != n) {
        if (input_matrix_[j] < 0) {
          return false;
        }
        j++;
      }
      if (input_matrix_[0] <= std::accumulate(input_matrix_.begin() + 1, input_matrix_.begin() + n - 1, 0)) {
        return false;
      }
    } else {
      j = 0;
      while (j != n) {
        if (input_matrix_[i * n + j] < 0) {
          return false;
        }
        j++;
      }
      if (i > 0 && i < n - 1) {
        if (input_matrix_[i * (n + 1)] <=
            std::accumulate(input_matrix_.begin() + i * n, input_matrix_.begin() + i * (n + 1) - 1, 0) +
                std::accumulate(input_matrix_.begin() + i * (n + 1) + 1, input_matrix_.begin() + (i + 1) * n - 1, 0)) {
          return false;
        }
      }
      if (i == n - 1) {
        if (input_matrix_[i * (n + 1)] <= std::accumulate(input_matrix_.begin() + i * n, input_matrix_.end() - 1, 0)) {
          return false;
        }
      }
    }
    i++;
  }
  return taskData->outputs_count[0] == 1;
}

bool deryabin_m_jacobi_iterative_method_seq::JacobiIterativeTaskSequential::run() {
  internal_order_test();
  unsigned short Nmax = 10000;
  unsigned short num_of_iterations = 0;
  double epsilon = pow(10, -6);
  double max_delta_x_i = 0;
  std::vector<double> x_old;
  auto n = (unsigned short)(input_right_vector_.size());
  do {
    x_old = output_x_vector_;
    unsigned short i = 0;
    unsigned short j;
    double sum;
    while (i != n) {
      j = 0;
      sum = 0;
      while (j != n) {
        if (i != j) {
          sum += input_matrix_[i * n + j] * x_old[j];
        }
        j++;
      }
      output_x_vector_[i] = (input_right_vector_[i] - sum) * (1.0 / input_matrix_[i * (n + 1)]);
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
