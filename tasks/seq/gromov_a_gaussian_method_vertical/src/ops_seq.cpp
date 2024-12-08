#include "seq/gromov_a_gaussian_method_vertical/include/ops_seq.hpp"

int gromov_a_gaussian_method_vertical_seq::matrix_rank(std::vector<double>& matrix, int rows, int columns,
                                                       int band_width) {
  int rank = 0;

  for (int i = 0; i < rows; ++i) {
    double max_element = 0.0;
    int max_row = i;

    for (int k = i; k < std::min(i + band_width, rows); ++k) {
      if (std::abs(matrix[k * columns + i]) > max_element) {
        max_element = std::abs(matrix[k * columns + i]);
        max_row = k;
      }
    }

    if (max_element == 0) {
      continue;
    }

    for (int k = 0; k < columns; ++k) {
      std::swap(matrix[max_row * columns + k], matrix[i * columns + k]);
    }

    for (int k = i + 1; k < std::min(i + band_width, rows); ++k) {
      double factor = matrix[k * columns + i] / matrix[i * columns + i];
      for (int j = i; j < columns; ++j) {
        matrix[k * columns + j] -= factor * matrix[i * columns + j];
      }
    }
    rank++;
  }
  return rank;
}

bool gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential::pre_processing() {
  internal_order_test();
  equations = taskData->inputs_count[1];

  input_coefficient.assign(taskData->inputs_count[0], 0);
  std::copy(reinterpret_cast<int*>(taskData->inputs[0]),
            reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0], input_coefficient.begin());

  input_rhs.assign(taskData->inputs_count[1], 0);
  std::copy(reinterpret_cast<int*>(taskData->inputs[1]),
            reinterpret_cast<int*>(taskData->inputs[1]) + taskData->inputs_count[1], input_rhs.begin());

  res.resize(equations);

  return true;
}

bool gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential::validation() {
  internal_order_test();
  std::vector<double> matrix_argument(equations * (equations + 1));

  std::copy(input_coefficient.begin(), input_coefficient.end(), matrix_argument.begin());
  std::copy(input_rhs.begin(), input_rhs.end(), matrix_argument.begin() + equations * equations);

  int rank_coeffs = matrix_rank(matrix_argument, equations, equations, band_width);
  int rank_augmented = matrix_rank(matrix_argument, equations, equations + 1, band_width);

  return (rank_coeffs == rank_augmented);
}

bool gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential::run() {
  internal_order_test();
  std::vector<double> matrix_argument(equations * (equations + 1));

  for (int i = 0; i < equations; ++i) {
    for (int j = 0; j < equations; ++j) {
      matrix_argument[i * (equations + 1) + j] = static_cast<double>(input_coefficient[i * equations + j]);
    }
    matrix_argument[i * (equations + 1) + equations] = static_cast<double>(input_rhs[i]);
  }

  for (int i = 0; i < equations; ++i) {
    double max_element = std::abs(matrix_argument[i * (equations + 1) + i]);
    int max_row = i;

    for (int k = i + 1; k < std::min(i + band_width, equations); ++k) {
      if (std::abs(matrix_argument[k * (equations + 1) + i]) > max_element) {
        max_element = std::abs(matrix_argument[k * (equations + 1) + i]);
        max_row = k;
      }
    }

    for (int j = 0; j <= equations; ++j) {
      std::swap(matrix_argument[max_row * (equations + 1) + j], matrix_argument[i * (equations + 1) + j]);
    }

    for (int k = i + 1; k < std::min(i + band_width, equations); ++k) {
      double factor = matrix_argument[k * (equations + 1) + i] / matrix_argument[i * (equations + 1) + i];
      for (int j = i; j <= equations; ++j) {
        matrix_argument[k * (equations + 1) + j] -= factor * matrix_argument[i * (equations + 1) + j];
      }
    }
  }

  for (int i = equations - 1; i >= 0; --i) {
    res[i] = matrix_argument[i * (equations + 1) + equations];
    for (int j = i + 1; j < equations; ++j) {
      res[i] -= matrix_argument[i * (equations + 1) + j] * res[j];
    }
    res[i] /= matrix_argument[i * (equations + 1) + i];
  }
  return true;
}

bool gromov_a_gaussian_method_vertical_seq::GaussVerticalSequential::post_processing() {
  internal_order_test();
  std::copy(res.begin(), res.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}