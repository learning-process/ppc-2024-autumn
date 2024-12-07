#include "seq/vasenkov_a_gauss_jordan_method_seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

void assignMatrixValues(std::vector<double>& matrix, const std::vector<double>& source,
                        int n, int k, int startRow, int endRow) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = 0; j < (n - k + 1); j++) {
            matrix[i * (n + 1) + k + j] = source[(i - startRow) * (n - k + 1) + j];
        }
    }
}


std::vector<double> vasenkov_a_gauss_jordan_method_seq::processMatrix(int n, int k, const std::vector<double>& matrix) {
    if (n <= 0 || k < 0 || k >= n) {
        throw std::invalid_argument("Invalid dimensions for matrix processing.");
    }

    int stride = n + 1;
    std::vector<double> result_vec(n * (n - k + 1));

    for (int i = 0; i < (n - k + 1); i++) {
        result_vec[i] = matrix[stride * k + k + i];
    }

    assignMatrixValues(result_vec, matrix, n, k, 0, k);
    assignMatrixValues(result_vec, matrix, n, k, k + 1, n);

    return result_vec;
}

void vasenkov_a_gauss_jordan_method_seq::updateMatrix(int n, int k, std::vector<double>& matrix,
                                            const std::vector<double>& iter_result) {
    int stride = n + 1;

    if (matrix.size() < (long unsigned int)(n * stride)) {
        throw std::out_of_range("Matrix size is insufficient.");
    }

    assignMatrixValues(matrix, iter_result, n, k, 0, k);
    assignMatrixValues(matrix, iter_result, n, k, k + 1, n);

    matrix[k * stride + k] /= matrix[k * stride + k];
    for (int i = 0; i < n; i++) {
        if (i != k) {
            matrix[i * stride + k] = 0;
        }
    }
}


bool vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential::validation() {
  internal_order_test();

  int n_val = *reinterpret_cast<int*>(taskData->inputs[1]);
  int matrix_size = taskData->inputs_count[0];
  return n_val * (n_val + 1) == matrix_size;
}

bool vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential::pre_processing() {
  internal_order_test();

  auto* matrix_data = reinterpret_cast<double*>(taskData->inputs[0]);
  int matrix_size = taskData->inputs_count[0];
  n = *reinterpret_cast<int*>(taskData->inputs[1]);
  matrix.assign(matrix_data, matrix_data + matrix_size);

  return true;
}

bool vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential::run() {
    internal_order_test();

    for (int k = 0; k < n; k++) {
        if (std::abs(matrix[k * (n + 1) + k]) < 1e-9) {
            bool swapped = false;
            for (int change = k + 1; change < n; change++) {
                if (std::abs(matrix[change * (n + 1) + k]) > 1e-9) {
                    for (int col = 0; col < (n + 1); col++) {
                        std::swap(matrix[k * (n + 1) + col], matrix[change * (n + 1) + col]);
                    }
                    swapped = true;
                    break;
                }
            }
            if (!swapped) return false;
        }

        std::vector<double> iter_matrix = vasenkov_a_gauss_jordan_method_seq::processMatrix(n, k, matrix);
        std::vector<double> iter_result((n - 1) * (n - k));

        int ind = 0;
        for (int i = 1; i < n; ++i) {
            double rel = iter_matrix[0];
            for (int j = 1; j < n - k + 1; ++j) {
                double nel = iter_matrix[i * (n - k + 1) + j];
                double a = iter_matrix[j];
                double b = iter_matrix[i * (n - k + 1)];
                iter_result[ind++] = nel - (a * b) / rel;
            }
        }

        vasenkov_a_gauss_jordan_method_seq::updateMatrix(n, k, matrix, iter_result);
    }

    return true;
}


bool vasenkov_a_gauss_jordan_method_seq::GaussJordanSequential::post_processing() {
  internal_order_test();

  auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(matrix.begin(), matrix.end(), output_data);

  return true;
}