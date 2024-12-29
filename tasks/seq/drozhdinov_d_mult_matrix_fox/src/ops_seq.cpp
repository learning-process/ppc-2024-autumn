// Copyright 2024 Nesterov Alexander
#include "seq/drozhdinov_d_mult_matrix_fox/include/ops_seq.hpp"

using namespace std::chrono_literals;

void drozhdinov_d_mult_matrix_fox_seq::SimpleMult(const std::vector<double>& A, const std::vector<double>& B,
                                                  std::vector<double>& C, int block) {
  for (int i = 0; i < block; ++i) {
    for (int j = 0; j < block; ++j) {
      for (int k = 0; k < block; ++k) {
        C[i * block + j] += A[i * block + k] * B[k * block + j];
      }
    }
  }
}

std::vector<double> drozhdinov_d_mult_matrix_fox_seq::paddingMatrix(const std::vector<double>& mat, int rows, int cols,
                                                                    int padding) {
  std::vector<double> padded(padding * padding, 0.0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      padded[i * padding + j] = mat[i * cols + j];
    }
  }
  return padded;
}

std::vector<double> drozhdinov_d_mult_matrix_fox_seq::SequentialFox(const std::vector<double>& A,
                                                                    const std::vector<double>& B, int k, int l, int n) {
  int C_rows = k;
  int C_cols = n;

  int padding = 1;
  while (padding < std::max({k, l, n})) {
    padding *= 2;
  }

  auto squareA = paddingMatrix(A, k, l, padding);
  auto squareB = paddingMatrix(B, l, n, padding);
  std::vector<double> squareC(padding * padding, 0.0);

  int grid_size = padding;

  int block_size = 1;

  std::vector<double> block_A(block_size * block_size);
  std::vector<double> block_B(block_size * block_size);
  std::vector<double> block_C(block_size * block_size, 0.0);

  for (int step = 0; step < grid_size; ++step) {
    for (int row = 0; row < grid_size; ++row) {
      for (int col = 0; col < grid_size; ++col) {
        int pivot = (row + step) % grid_size;

        for (int i = 0; i < block_size; ++i) {
          for (int j = 0; j < block_size; ++j) {
            block_A[i * block_size + j] = squareA[(row * block_size + i) * padding + (pivot * block_size + j)];
            block_B[i * block_size + j] = squareB[(pivot * block_size + i) * padding + (col * block_size + j)];
          }
        }

        SimpleMult(block_A, block_B, block_C, block_size);

        for (int i = 0; i < block_size; ++i) {
          for (int j = 0; j < block_size; ++j) {
            squareC[(row * block_size + i) * padding + (col * block_size + j)] += block_C[i * block_size + j];
          }
        }

        std::fill(block_C.begin(), block_C.end(), 0.0);
      }
    }
  }

  std::vector<double> C(C_rows * C_cols);
  for (int i = 0; i < C_rows; ++i) {
    for (int j = 0; j < C_cols; ++j) {
      C[i * C_cols + j] = squareC[i * padding + j];
    }
  }

  return C;
}

bool drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  k = taskData->inputs_count[0];
  l = taskData->inputs_count[1];
  m = taskData->inputs_count[2];
  n = taskData->inputs_count[3];
  A.resize(k * l);
  B.resize(m * n);
  auto* ptra = reinterpret_cast<double*>(taskData->inputs[0]);
  for (int i = 0; i < k * l; i++) {
    A[i] = ptra[i];
  }
  auto* ptrb = reinterpret_cast<double*>(taskData->inputs[1]);
  for (int i = 0; i < m * n; i++) {
    B[i] = ptrb[i];
  }
  return true;
}

bool drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[1] == taskData->inputs_count[2] && taskData->inputs.size() == 2 &&
         taskData->outputs.size() == 1 && taskData->outputs_count[0] == taskData->inputs_count[0] &&
         taskData->outputs_count[1] == taskData->inputs_count[3];
}

bool drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential::run() {
  internal_order_test();
  C = drozhdinov_d_mult_matrix_fox_seq::SequentialFox(A, B, k, l, n);
  return true;
}

bool drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < k * n; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = C[i];
  }
  return true;
}
