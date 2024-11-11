#include "seq/belov_a_gauss_seidel_iter_method/include/ops_seq.hpp"

#include <random>

using namespace std;

namespace belov_a_gauss_seidel_seq {

bool GaussSeidelSequential::isDiagonallyDominant() const {
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

bool GaussSeidelSequential::pre_processing() {
  internal_order_test();

  n = taskData->inputs_count[0];
  auto* inputMatrixData = reinterpret_cast<double*>(taskData->inputs[0]);
  A.assign(inputMatrixData, inputMatrixData + n * n);

  auto* freeMembersVector = reinterpret_cast<double*>(taskData->inputs[1]);
  b.assign(freeMembersVector, freeMembersVector + n);

  epsilon = *(reinterpret_cast<double*>(taskData->inputs[2]));

  x.assign(n, 0.0);  // initial approximations

  cout << "Input matrix: { ";
  for (const auto& item : A) {
    cout << item << " ";
  }
  cout << "}" << endl;

  cout << "Free members column: { ";
  for (const auto& item : b) {
    cout << item << " ";
  }
  cout << "}" << endl;

  return true;
}

bool GaussSeidelSequential::validation() {
  internal_order_test();

  return (taskData->inputs.size() == 3 && !taskData->inputs_count.empty() && !taskData->outputs.empty() &&
          (taskData->inputs_count[0] * taskData->inputs_count[0] == taskData->inputs_count[1]));
}

bool GaussSeidelSequential::run() {
  internal_order_test();

  if (!isDiagonallyDominant()) return false;

  vector<double> x_new(n, 0.0);
  double norm;
  int iter = 0;

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

      norm += pow(x_new[i] - x[i], 2);
    }

    for (int i = 0; i < n; ++i) {
      norm += (x_new[i] - x[i]) * (x_new[i] - x[i]);
      x[i] = x_new[i];
    }
    norm = sqrt(norm);
    iter++;

  } while (norm > epsilon);

  return true;
}

bool GaussSeidelSequential::post_processing() {
  internal_order_test();

  cout << "X-column: { ";
  for (const auto& item : x) {
    cout << item << " ";
  }
  cout << "}" << endl;
  copy(x.begin(), x.end(), reinterpret_cast<double*>(taskData->outputs[0]));
  return true;
}

vector<double> generateDiagonallyDominantMatrix(int n) {
  vector<double> A_local(n * n, 0.0);

  random_device rd;
  mt19937 gen(rd());
  uniform_real_distribution<> dis(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A_local[i * n + j] = dis(gen);
        row_sum += abs(A_local[i * n + j]);
      }
    }
    A_local[i * n + i] = row_sum + abs(dis(gen)) + 1.0;
  }
  return A_local;
}

}  // namespace belov_a_gauss_seidel_seq
