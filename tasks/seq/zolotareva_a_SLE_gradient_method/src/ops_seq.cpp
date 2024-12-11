// Copyright 2024 Nesterov Alexander
#include "seq/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
using namespace std;

bool zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::validation() {
  internal_order_test();
  n_ = static_cast<int>(taskData->inputs_count[1]);
  if (taskData->inputs_count.size() < 2 || taskData->inputs.size() < 2 || taskData->outputs.size() < 1) {
    return false;
  }
  if (int(taskData->inputs_count[0]) != n_ * n_ || int(taskData->inputs_count[1]) != n_) {
    return false;
  }
  if (int(taskData->outputs_count[0]) != n_) {
    return false;
  }
  // проверка симметрии
  const double* A = reinterpret_cast<double*>(taskData->inputs[0]);
  for (int i = 0; i < n_; ++i) {
    for (int j = i + 1; j < n_; ++j) {
      if (A[i * n_ + j] != A[j * n_ + i]) {
        return false;
      }
    }
  }
  return true;
}

bool zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  A_.resize(n_ * n_);
  b_.resize(n_);
  x_.resize(n_, 0.0);
  const double* input_matrix = reinterpret_cast<const double*>(taskData->inputs[0]);
  const double* input_vector = reinterpret_cast<const double*>(taskData->inputs[1]);

  for (int i = 0; i < n_; ++i) {
    b_[i] = input_vector[i];
    for (int j = 0; j < n_; ++j) {
      A_[i * n_ + j] = input_matrix[i * n_ + j];
    }
  }

  return true;
}

bool zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::run() {
  internal_order_test();
  x_ = conjugate_gradient(A_, b_, x_, n_);
  return true;
}

bool zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* output_raw = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(x_.begin(), x_.end(), output_raw);
  return true;
}

std::vector<double> zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::conjugate_gradient(
    const std::vector<double>& A, const std::vector<double>& b, std::vector<double>& x, int N) {
  double threshold = sqrt(dot_product(b, b, N));
  threshold = threshold * 1e-12;
  if (threshold == 0) {
    threshold = 1e-12;
  }
  std::vector<double> r = b;  // Начальный вектор невязки r = b - A*x0, x0 = 0
  std::vector<double> p = r;  // Начальное направление поиска p = r
  double rs_old = dot_product(r, r, N);

  for (int s = 0; s <= N; ++s) {
    std::vector<double> Ap = matrix_vector_mult(A, p, N);
    if (dot_product(p, Ap, N) == 0) break;

    double alpha = rs_old / dot_product(p, Ap, N);

    for (int i = 0; i < N; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double rs_new = dot_product(r, r, N);
    if (rs_new < threshold) {  // Проверка на сходимость
      break;
    }
    double beta = rs_new / rs_old;
    for (int i = 0; i < N; ++i) {
      p[i] = r[i] + beta * p[i];
    }

    rs_old = rs_new;
  }

  return x;
}

double zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::dot_product(const std::vector<double>& vec1,
                                                                             const std::vector<double>& vec2, int n) {
  double sum = 0.0;
  for (int i = 0; i < n; ++i) {
    sum += vec1[i] * vec2[i];
  }
  return sum;
}

std::vector<double> zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::matrix_vector_mult(
    const std::vector<double>& matrix, const std::vector<double>& vector, int n) {
  std::vector<double> result(n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result[i] += matrix[i * n + j] * vector[j];
    }
  }
  return result;
}

/*bool zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::is_positive_definite(const std::vector<double>& A,
                                                                                    int n) {
  std::vector<float> diag(n, 0.0f);

  return true;
}*/