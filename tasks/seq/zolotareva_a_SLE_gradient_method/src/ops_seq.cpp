#include "seq/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
using namespace std;

bool zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::validation() {
  internal_order_test();

  if (static_cast<int>(taskData->inputs_count[0]) < 0 || static_cast<int>(taskData->inputs_count[1]) < 0 ||
      static_cast<int>(taskData->outputs_count[0]) < 0)
    return false;
  if (taskData->inputs_count.size() < 2 || taskData->inputs.size() < 2 || taskData->outputs.empty()) return false;

  if (static_cast<int>(taskData->inputs_count[0]) !=
      (static_cast<int>(taskData->inputs_count[1]) * static_cast<int>(taskData->inputs_count[1])))
    return false;

  if (taskData->outputs_count[0] != taskData->inputs_count[1]) return false;

  // проверка симметрии и положительной определённости
  const auto* A = reinterpret_cast<const double*>(taskData->inputs[0]);

  return is_positive_and_simm(A, static_cast<int>(taskData->inputs_count[1]));
}

bool zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  n_ = static_cast<int>(taskData->inputs_count[1]);
  A_.resize(n_ * n_);
  b_.resize(n_);
  x_.resize(n_, 0.0);
  const auto* input_matrix = reinterpret_cast<const double*>(taskData->inputs[0]);
  const auto* input_vector = reinterpret_cast<const double*>(taskData->inputs[1]);

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
  conjugate_gradient(A_, b_, x_, n_);
  return true;
}

bool zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* output_raw = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(x_.begin(), x_.end(), output_raw);
  return true;
}

void zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::conjugate_gradient(const std::vector<double>& A,
                                                                                  const std::vector<double>& b,
                                                                                  std::vector<double>& x, int N) {
  double initial_res_norm = 0.0;
  dot_product(initial_res_norm, b, b, N);
  initial_res_norm = std::sqrt(initial_res_norm);
  double threshold = initial_res_norm == 0.0 ? 1e-12 : (1e-12 * initial_res_norm);

  std::vector<double> r = b;  // начальный вектор невязки r = b - A*x0, x0 = 0
  std::vector<double> p = r;  // начальное направление поиска p = r
  double rs_old = 0;
  dot_product(rs_old, r, r, N);

  for (int s = 0; s <= N; ++s) {
    std::vector<double> Ap(N, 0.0);
    matrix_vector_mult(A, p, Ap, N);
    double pAp = 0.0;
    dot_product(pAp, p, Ap, N);
    if (pAp == 0.0) break;

    double alpha = rs_old / pAp;

    for (int i = 0; i < N; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * Ap[i];
    }

    double rs_new = 0.0;
    dot_product(rs_new, r, r, N);
    if (rs_new < threshold) {  // Проверка на сходимость
      break;
    }
    double beta = rs_new / rs_old;
    for (int i = 0; i < N; ++i) {
      p[i] = r[i] + beta * p[i];
    }

    rs_old = rs_new;
  }
}

void zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::dot_product(double& sum, const std::vector<double>& vec1,
                                                                           const std::vector<double>& vec2, int n) {
  for (int i = 0; i < n; ++i) {
    sum += vec1[i] * vec2[i];
  }
}

void zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::matrix_vector_mult(const std::vector<double>& matrix,
                                                                                  const std::vector<double>& vector,
                                                                                  std::vector<double>& result, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      result[i] += matrix[i * n + j] * vector[j];
    }
  }
}

bool zolotareva_a_SLE_gradient_method_seq::TestTaskSequential::is_positive_and_simm(const double* A, int n) {
  std::vector<double> M(n * n);
  // копируем и проверяем симметричность
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double val = A[i * n + j];
      M[i * n + j] = val;
      if (j > i) {
        if (val != A[j * n + i]) {
          return false;
        }
      }
    }
  }
  // проверяем позитивную определенность
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j <= i; ++j) {
      double sum = M[i * n + j];
      for (int k = 0; k < j; k++) {
        sum -= M[i * n + k] * M[j * n + k];
      }

      if (i == j) {
        if (sum <= 1e-15) {
          return false;
        }
        M[i * n + j] = std::sqrt(sum);
      } else {
        M[i * n + j] = sum / M[j * n + j];
      }
    }
  }
  return true;
}