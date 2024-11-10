#include "seq/kolokolova_d_gaussian_method_horizontal/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

int kolokolova_d_gaussian_method_horizontal_seq::find_rank(std::vector<double>& matrix, int rows, int cols) {
  int rank = 0;

  for (int i = 0; i < rows; ++i) {
    // Find max element
    double max_elem = 0.0;
    int max_row = i;
    for (int k = i; k < rows; ++k) {
      if (std::abs(matrix[k * cols + i]) > max_elem) {
        max_elem = std::abs(matrix[k * cols + i]);
        max_row = k;
      }
    }

    // If all matrice is 0, than rank = 0
    if (max_elem == 0) {
      continue;
    }

    // Rearranging rows to move the max element to the current position
    for (int k = 0; k < cols; ++k) {
      std::swap(matrix[max_row * cols + k], matrix[i * cols + k]);
    }

    // Make all elements below the current to zero
    for (int k = i + 1; k < rows; ++k) {
      double factor = matrix[k * cols + i] / matrix[i * cols + i];
      for (int j = i; j < cols; ++j) {
        matrix[k * cols + j] -= factor * matrix[i * cols + j];
      }
    }

    rank++;
  }
  return rank;
}

bool kolokolova_d_gaussian_method_horizontal_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  count_equations = taskData->inputs_count[1];

  // Init value for input and output
  input_coeff = std::vector<int>(taskData->inputs_count[0]);
  auto* tmp_ptr_coeff = reinterpret_cast<int*>(taskData->inputs[0]);
  for (unsigned i = 0; i < taskData->inputs_count[0]; i++) {
    input_coeff[i] = tmp_ptr_coeff[i];
  }

  input_y = std::vector<int>(taskData->inputs_count[1]);
  auto* tmp_ptr_y = reinterpret_cast<int*>(taskData->inputs[1]);
  for (unsigned i = 0; i < taskData->inputs_count[1]; i++) {
    input_y[i] = tmp_ptr_y[i];
  }
  res.resize(count_equations);
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_seq::TestTaskSequential::validation() {
  internal_order_test();

  std::vector<double> matrix_argum(count_equations * (count_equations + 1));

  // Filling the matrix
  for (int i = 0; i < count_equations; ++i) {
    for (int j = 0; j < count_equations; ++j) {
      matrix_argum[i * (count_equations + 1) + j] = static_cast<double>(input_coeff[i * count_equations + j]);
    }
    matrix_argum[i * (count_equations + 1) + count_equations] = static_cast<double>(input_y[i]);
  }

  // Get rangs of matrices
  int rank_A = find_rank(matrix_argum, count_equations, count_equations);
  int rank_Ab = find_rank(matrix_argum, count_equations, count_equations + 1);

  // Checking for inconsistency
  if (rank_A != rank_Ab) {
    return false;
  }
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_seq::TestTaskSequential::run() {
  internal_order_test();
  std::vector<double> matrix_argum(count_equations * (count_equations + 1));
  // Filling the matrix
  for (int i = 0; i < count_equations; ++i) {
    for (int j = 0; j < count_equations; ++j) {
      matrix_argum[i * (count_equations + 1) + j] = static_cast<double>(input_coeff[i * count_equations + j]);
    }
    matrix_argum[i * (count_equations + 1) + count_equations] = static_cast<double>(input_y[i]);
  }

  // Forward Gaussian move
  for (int i = 0; i < count_equations; ++i) {
    // Find max of element
    double max_elem = std::abs(matrix_argum[i * (count_equations + 1) + i]);
    int max_row = i;
    for (int k = i + 1; k < count_equations; ++k) {
      if (std::abs(matrix_argum[k * (count_equations + 1) + i]) > max_elem) {
        max_elem = std::abs(matrix_argum[k * (count_equations + 1) + i]);
        max_row = k;
      }
    }
    for (int j = 0; j <= count_equations; ++j) {
      std::swap(matrix_argum[max_row * (count_equations + 1) + j], matrix_argum[i * (count_equations + 1) + j]);
    }

    // Division by max element and subtraction
    for (int k = i + 1; k < count_equations; ++k) {
      double factor = matrix_argum[k * (count_equations + 1) + i] / matrix_argum[i * (count_equations + 1) + i];
      for (int j = i; j <= count_equations; ++j) {
        matrix_argum[k * (count_equations + 1) + j] -= factor * matrix_argum[i * (count_equations + 1) + j];
      }
    }
  }

  // Gaussian reversal
  for (int i = count_equations - 1; i >= 0; --i) {
    res[i] = matrix_argum[i * (count_equations + 1) + count_equations];
    for (int j = i + 1; j < count_equations; ++j) {
      res[i] -= matrix_argum[i * (count_equations + 1) + j] * res[j];
    }
    res[i] /= matrix_argum[i * (count_equations + 1) + i];
  }
  return true;
}

bool kolokolova_d_gaussian_method_horizontal_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < count_equations; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = res[i];
  }
  return true;
}
