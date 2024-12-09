#include "seq/kavtorev_d_iterative_jacobi/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <thread>

bool kavtorev_d_iterative_jacobi_seq::jacobi_method(int N, const std::vector<double>& A_flat,
                                                    const std::vector<double>& F, std::vector<double>& X, double eps,
                                                    int iterations) {
  std::vector<double> TempX(N);
  double norm;

  int iteration = 0;
  do {
    for (int i = 0; i < N; i++) {
      TempX[i] = F[i];
      for (int j = 0; j < N; j++) {
        if (i != j) TempX[i] -= A_flat[i * N + j] * X[j];
      }
      TempX[i] /= A_flat[i * N + i];
    }

    norm = std::fabs(X[0] - TempX[0]);
    for (int i = 0; i < N; i++) {
      double diff = std::fabs(X[i] - TempX[i]);
      if (diff > norm) norm = diff;
      X[i] = TempX[i];
    }
    iteration++;
  } while (iteration < iterations && norm > eps);
  return iteration != iterations;
}

bool kavtorev_d_iterative_jacobi_seq::IterativeJacobiSequential::validation() {
  internal_order_test();

  if (!taskData) {
    return false;
  }

  if (taskData->inputs.size() < 5 || taskData->outputs.empty()) {
    return false;
  }

  if ((taskData->inputs[0] == nullptr) || (taskData->inputs[1] == nullptr) || (taskData->inputs[2] == nullptr) ||
      (taskData->inputs[3] == nullptr) || (taskData->inputs[4] == nullptr)) {
    return false;
  }

  if (taskData->outputs[0] == nullptr) {
    return false;
  }

  int val_n = *reinterpret_cast<int*>(taskData->inputs[0]);

  if (val_n <= 0) {
    return false;
  }

  double val_eps = *reinterpret_cast<double*>(taskData->inputs[1]);

  if (val_eps <= 0.0) {
    return false;
  }

  int val_iterations = *reinterpret_cast<int*>(taskData->inputs[2]);

  if (val_iterations <= 0) {
    return false;
  }

  auto* val_A_data = reinterpret_cast<double*>(taskData->inputs[3]);
  int val_A_size = taskData->inputs_count[3];

  auto* val_F_data = reinterpret_cast<double*>(taskData->inputs[4]);
  int val_F_size = taskData->inputs_count[4];

  auto* val_output = reinterpret_cast<double*>(taskData->outputs[0]);
  int val_output_size = taskData->outputs_count[0];

  if ((val_A_data == nullptr) || (val_F_data == nullptr) || (val_output == nullptr)) {
    return false;
  }

  if (val_A_size != val_n * val_n) {
    return false;
  }

  if (val_F_size != val_n) {
    return false;
  }

  if (val_output_size < val_n) {
    return false;
  }

  std::vector<double> val_A_flat(val_A_data, val_A_data + val_A_size);
  std::vector<double> val_F(val_F_data, val_F_data + val_F_size);

  for (size_t i = 0; i < val_A_flat.size(); ++i) {
    if (std::isnan(val_A_flat[i]) || std::isinf(val_A_flat[i])) {
      return false;
    }
  }

  for (size_t i = 0; i < val_F.size(); ++i) {
    if (std::isnan(val_F[i]) || std::isinf(val_F[i])) {
      return false;
    }
  }

  for (int i = 0; i < val_n; ++i) {
    double diagonal_element = val_A_flat[i * val_n + i];
    if (diagonal_element == 0.0) {
      return false;
    }
  }

  for (int i = 0; i < val_n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < val_n; ++j) {
      if (j != i) {
        sum += std::abs(val_A_flat[i * val_n + j]);
      }
    }
    double diagonal_element = std::abs(val_A_flat[i * val_n + i]);
    if (diagonal_element < sum) {
      return false;
    }
  }

  return true;
}

bool kavtorev_d_iterative_jacobi_seq::IterativeJacobiSequential::pre_processing() {
  internal_order_test();

  n = *reinterpret_cast<int*>(taskData->inputs[0]);
  eps = *reinterpret_cast<double*>(taskData->inputs[1]);
  iterations = *reinterpret_cast<int*>(taskData->inputs[2]);
  auto* A_data = reinterpret_cast<double*>(taskData->inputs[3]);
  int A_size = taskData->inputs_count[3];
  auto* F_data = reinterpret_cast<double*>(taskData->inputs[4]);
  int F_size = taskData->inputs_count[4];

  try {
    A_flat.assign(A_data, A_data + A_size);
    F.assign(F_data, F_data + F_size);
    result_vector.resize(n);
  } catch (const std::exception& e) {
    std::cerr << "Error during global data assignment: " << e.what() << std::endl;
    return false;
  }

  return true;
}

bool kavtorev_d_iterative_jacobi_seq::IterativeJacobiSequential::run() {
  internal_order_test();

  return kavtorev_d_iterative_jacobi_seq::jacobi_method(n, A_flat, F, result_vector, eps, iterations);
}

bool kavtorev_d_iterative_jacobi_seq::IterativeJacobiSequential::post_processing() {
  internal_order_test();

  auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(result_vector.begin(), result_vector.end(), output_data);

  return true;
}
