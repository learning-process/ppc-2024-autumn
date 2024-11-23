// Copyright 2024 Nesterov Alexander
#include "seq/drozhdinov_d_gauss_vertical_scheme/include/ops_seq.hpp"
// not example
#include <random>
#include <thread>

using namespace std::chrono_literals;

int mkLinCoordddm(int x, int y, int xSize) { return y * xSize + x; }

std::vector<double> genElementaryMatrix(int rows, int columns) {
  std::vector<double> res;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (i == j) {
        res.push_back(1);
      } else {
        res.push_back(0);
      }
    }
  }
  return res;
}

std::vector<double> genDenseMatrix(int n, int a) {
  std::vector<double> dense;
  std::vector<double> ed(n * n);
  std::vector<double> res(n * n);
  for (int i = 0; i < n; i++) {
    for (int j = i; j < n + i; j++) {
      dense.push_back(a + j);
    }
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i < 2) {
        ed[j * n + i] = 0;
      } else if (i == j && i >= 2) {
        ed[j * n + i] = 1;
      } else {
        ed[j * n + i] = 0;
      }
    }
  }
  for (int i = 0; i < n * n; i++) {
    res[i] = (dense[i] + ed[i]);
  }
  return res;
}

std::vector<int> drozhdinov_d_gauss_vertical_scheme_seq::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

double myrnd(double value) { return (fabs(value - std::round(value)) < GAMMA ? std::round(value) : value); }

int drozhdinov_d_gauss_vertical_scheme_seq::Myrank(std::vector<double> matrix, int m, int n) {
  int row = 0;
  int col = 0;

  while (row < m && col < n) {
    int max_row = row;
    for (int i = row + 1; i < m; i++) {
      if (abs(matrix[i * n + col]) > abs(matrix[max_row * n + col])) {
        max_row = i;
      }
    }

    if (abs(matrix[max_row * n + col]) < 1e-9) {
      col++;
    } else {
      if (max_row != row) {
        for (int j = col; j < n; j++) {
          std::swap(matrix[row * n + j], matrix[max_row * n + j]);
        }
      }

      for (int i = row + 1; i < m; i++) {
        double factor = matrix[i * n + col] / matrix[row * n + col];
        for (int j = col; j < n; j++) {
          matrix[i * n + j] -= factor * matrix[row * n + j];
        }
      }
      row++;
      col++;
    }
  }

  int rank = 0;
  for (int i = 0; i < m; i++) {
    bool is_nonzero = false;
    for (int j = 0; j < n; j++) {
      if (abs(matrix[i * n + j]) > 1e-9) {
        is_nonzero = true;
        break;
      }
    }
    if (is_nonzero) {
      rank++;
    }
  }

  return rank;
}

std::vector<double> drozhdinov_d_gauss_vertical_scheme_seq::extendedMatrix(const std::vector<double> A, int n,
                                                                           const std::vector<double> b) {
  std::vector<double> extendedMatrix(n * (n + 1));

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      extendedMatrix[i * (n + 1) + j] = A[i * n + j];
    }
    extendedMatrix[i * (n + 1) + n] = b[i];
  }
  return extendedMatrix;
}

double drozhdinov_d_gauss_vertical_scheme_seq::Determinant(const std::vector<double>& matrix, int n) {
  std::vector<double> tempMatrix = matrix;
  double det = 1.0;
  for (int i = 0; i < n; ++i) {
    int maxRow = i;
    for (int k = i + 1; k < n; ++k) {
      if (fabs(tempMatrix[k * n + i]) > fabs(tempMatrix[maxRow * n + i])) {
        maxRow = k;
      }
    }
    if (fabs(tempMatrix[maxRow * n + i]) < 1e-9) {
      return 0.0;
    }
    if (maxRow != i) {
      for (int k = 0; k < n; ++k) {
        std::swap(tempMatrix[i * n + k], tempMatrix[maxRow * n + k]);
      }
      det *= -1;
    }
    det *= tempMatrix[i * n + i];
    for (int k = i + 1; k < n; ++k) {
      tempMatrix[i * n + k] /= tempMatrix[i * n + i];
    }
    for (int k = i + 1; k < n; ++k) {
      for (int j = i + 1; j < n; ++j) {
        tempMatrix[k * n + j] -= tempMatrix[k * n + i] * tempMatrix[i * n + j];
      }
    }
  }
  return det;
}

bool drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  // Init value for input and output
  coefs = std::vector<double>(taskData->inputs_count[0]);
  auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    coefs[i] = ptr[i];
  }
  b = std::vector<double>(taskData->inputs_count[1]);
  auto* ptr1 = reinterpret_cast<double*>(taskData->inputs[1]);
  for (unsigned int i = 0; i < taskData->inputs_count[1]; i++) {
    b[i] = ptr1[i];
  }
  columns = taskData->inputs_count[2];
  rows = taskData->inputs_count[3];
  if ((Myrank(coefs, columns, rows) == Myrank(extendedMatrix(coefs, rows, b), rows + 1, rows)) ||
      Determinant(coefs, rows) == 0) {
    return false;
  }
  return true;
}

bool drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->inputs_count[3] == taskData->inputs_count[2] &&
          taskData->inputs_count[2] == taskData->outputs_count[0] && taskData->inputs.size() == 2 &&
          taskData->outputs.size() == 1;
}

bool drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential::run() {
  internal_order_test();
  std::vector<double> result(rows);
  std::vector<double> current(rows);
  for (int i = 0; i < rows; i++) {
    major.push_back(false);
    row_number.push_back(0);
  }
  for (int i = 0; i < columns; i++) {
    double max = 0;
    int index = 0;
    for (int j = 0; j < rows; j++) {
      if ((fabs(coefs[mkLinCoordddm(j, i, columns)]) >= fabs(max)) && (!major[j])) {
        max = coefs[mkLinCoordddm(j, i, columns)];
        index = j;
      }
    }
    major[index] = true;
    row_number[i] = index;
    for (int ii = 0; ii < rows; ii++) {
      current[ii] = 0;
      if (!major[ii]) {
        current[ii] = coefs[mkLinCoordddm(i, ii, columns)] / coefs[mkLinCoordddm(i, index, columns)];
      }
    }
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        if (!major[row]) {
          coefs[mkLinCoordddm(column, row, columns)] -= coefs[mkLinCoordddm(column, index, columns)] * current[row];
        }
      }
      if (!major[row]) {
        b[row] -= b[index] * current[row];
      }
    }
  }
  for (int k = 0; k < rows; k++) {
    if (!major[k]) {
      row_number[rows - 1] = k;
      break;
    }
  }
  for (int m = rows - 1; m >= 0; m--) {
    elem = 0;
    for (int n = m + 1; n < rows; n++) {
      elem += result[n] * coefs[mkLinCoordddm(n, row_number[m], columns)];
    }
    result[m] = myrnd((b[row_number[m]] - elem) / coefs[mkLinCoordddm(m, row_number[m], columns)]);
  }
  for (auto v : result) {
    x.push_back(v);
  }
  return true;
}

bool drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < columns; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x[i];
  }
  return true;
}
