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
  return true;
}

bool drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count[3] == taskData->inputs_count[2] &&
          taskData->inputs_count[2] == taskData->outputs_count[0]);
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
