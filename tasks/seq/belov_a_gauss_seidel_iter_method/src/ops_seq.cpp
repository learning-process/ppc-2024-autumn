#include "seq/belov_a_gauss_seidel_iter_method/include/ops_seq.hpp"

using namespace std;

namespace belov_a_gauss_seidel_seq {

bool isDiagonallyDominant(const vector<double>& A, int n) {
  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;

    for (int j = 0; j < n; ++j) {
      if (i != j) row_sum += abs(A[i * n + j]);
    }

    if (abs(A[i * n + i]) <= row_sum) {
      return false;
    }
  }

  return true;
}

bool GaussSeidelSequential::ge_double(double a, double b) {
  return (a - b) >= -((fabs(a) < fabs(b) ? fabs(b) : fabs(a)) * std::numeric_limits<double>::epsilon());
}

bool GaussSeidelSequential::pre_processing() {
  internal_order_test();

  n = taskData->inputs_count[0];
  auto* inputMatrixData = reinterpret_cast<double*>(taskData->inputs[0]);
  A.assign(inputMatrixData, inputMatrixData + n * n);

  auto* freeMembersVector = reinterpret_cast<double*>(taskData->inputs[1]);
  b.assign(freeMembersVector, freeMembersVector + n);

  epsilon = *(reinterpret_cast<double*>(taskData->inputs[2]));

  x.assign(n, 0.0);  // initial approximations

  return true;
}

bool GaussSeidelSequential::validation() {
  internal_order_test();

  if (taskData->inputs.size() != 3 || taskData->inputs_count.size() < 3 ||
      taskData->inputs_count[0] * taskData->inputs_count[0] != taskData->inputs_count[2])
    return false;

  vector<double> mt;
  auto* mt_data = reinterpret_cast<double*>(taskData->inputs[0]);
  mt.assign(mt_data, mt_data + taskData->inputs_count[0] * taskData->inputs_count[0]);

  return (!taskData->outputs.empty() && (taskData->inputs_count[0] == taskData->inputs_count[1]) &&
          isDiagonallyDominant(mt, taskData->inputs_count[0]));
}

bool GaussSeidelSequential::run() {
  internal_order_test();

  vector<double> x_new(n, 0.0);
  double norm;

  do {
    norm = 0.0;

    for (int i = 0; i < n; ++i) {
      double sum = 0.0;
      for (int j = 0; j < n; ++j) {
        if (i != j) {
          sum += A[i * n + j] * x[j];
        }
      }
      x_new[i] = (b[i] - sum) / A[i * n + i];
    }

    for (int i = 0; i < n; ++i) {
      norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
      x[i] = x_new[i];
    }
    norm = sqrt(norm);

  } while (ge_double(norm, epsilon));

  return true;
}

bool GaussSeidelSequential::post_processing() {
  internal_order_test();

  copy(x.begin(), x.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}

}  // namespace belov_a_gauss_seidel_seq
