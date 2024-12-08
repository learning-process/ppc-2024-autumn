// Filatev Vladislav Metod Zedela
#include "seq/filatev_v_metod_zedela/include/ops_seq.hpp"

filatev_v_metod_zedela_seq::MetodZedela::MetodZedela(std::shared_ptr<ppc::core::TaskData> taskData_)
    : Task(std::move(taskData_)) {
  this->size = taskData->inputs_count[0];
  this->delit.resize(size);

  auto* temp = reinterpret_cast<int*>(taskData->inputs[0]);
  this->matrix.insert(matrix.end(), temp, temp + size * size);

  temp = reinterpret_cast<int*>(taskData->inputs[1]);
  this->bVectrot.insert(bVectrot.end(), temp, temp + size);
}

bool filatev_v_metod_zedela_seq::MetodZedela::validation() {
  internal_order_test();

  if (taskData->inputs_count[0] != taskData->outputs_count[0]) {
    return false;
  }
  int rank = rankMatrix(matrix, size);
  if (rank == 0 || rank != rankRMatrix()) {
    return false;
  }
  if (determinant() == 0) {
    return false;
  }
  for (int i = 0; i < size; ++i) {
    int sum = 0;
    for (int j = 0; j < size; ++j) sum += abs(matrix[i * size + j]);
    sum -= abs(matrix[i * size + i]);
    if (sum > abs(matrix[i * size + i])) {
      return false;
    }
  }
  return true;
}

bool filatev_v_metod_zedela_seq::MetodZedela::pre_processing() {
  internal_order_test();
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      if (i == j) {
        delit[i] = matrix[i * size + j];
        matrix[i * size + j] = bVectrot[i];
      } else {
        matrix[i * size + j] *= -1;
      }
    }
  }
  return true;
}

bool filatev_v_metod_zedela_seq::MetodZedela::run() {
  internal_order_test();

  std::vector<double> it1(size, 0);
  std::vector<double> it2(size);
  double sum1 = 0;
  double sum2 = 0;
  double sum;

  do {
    std::swap(it1, it2);
    std::swap(sum1, sum2);
    sum1 = 0;
    for (int i = 0; i < size; ++i) {
      sum = 0;
      for (int j = 0; j < i; ++j) {
        sum += it1[j] * matrix[i * size + j];
      }
      sum += matrix[(size + 1) * i];
      for (int j = i + 1; j < size; ++j) {
        sum += it2[j] * matrix[i * size + j];
      }
      it1[i] = sum / delit[i];
      sum1 += it1[i];
    }
  } while (std::abs(sum1 - sum2) > alfa);

  answer = it1;

  return true;
}

bool filatev_v_metod_zedela_seq::MetodZedela::post_processing() {
  internal_order_test();
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(answer.data()));
  return true;
}

void filatev_v_metod_zedela_seq::MetodZedela::setAlfa(double _alfa) { this->alfa = _alfa / 100; }

int filatev_v_metod_zedela_seq::MetodZedela::rankMatrix(std::vector<int>& matrixT, int n) const {
  std::vector<double> _matrix(matrixT.size());
  std::transform(matrixT.begin(), matrixT.end(), _matrix.begin(), [](int val) { return static_cast<double>(val); });
  if (n == 0) return 0;
  int m = size;

  int rank = 0;

  for (int col = 0; col < n; col++) {
    int pivotRow = -1;
    for (int row = rank; row < m; row++) {
      if (_matrix[row * n + col] != 0) {
        pivotRow = row;
        break;
      }
    }

    if (pivotRow == -1) continue;

    if (rank != pivotRow)
      std::swap_ranges(_matrix.begin() + rank * n, _matrix.begin() + (rank + 1) * n, _matrix.begin() + pivotRow * n);

    for (int row = 0; row < m; row++) {
      if (row != rank) {
        double factor = _matrix[row * n + col] / _matrix[rank * n + col];
        for (int j = col; j < n; j++) {
          _matrix[row * n + j] -= factor * _matrix[rank * n + j];
        }
      }
    }
    rank++;
  }

  return rank;
}

double filatev_v_metod_zedela_seq::MetodZedela::determinant() {
  std::vector<double> L(size * size, 0.0);
  std::vector<double> U(size * size, 0.0);

  for (int i = 0; i < size; i++) {
    for (int j = i; j < size; j++) {
      U[i * size + j] = matrix[i * size + j];
      for (int k = 0; k < i; k++) {
        U[i * size + j] -= L[i * size + k] * U[k * size + j];
      }
    }

    for (int j = i + 1; j < size; j++) {
      L[j * size + i] = matrix[j * size + i];
      for (int k = 0; k < i; k++) {
        L[j * size + i] -= L[j * size + k] * U[k * size + i];
      }
      L[j * size + i] /= U[i * size + i];
    }

    L[i * size + i] = 1;
  }

  double det = 1.0;
  for (int i = 0; i < size; i++) {
    det *= U[i * size + i];
  }
  return det;
}

int filatev_v_metod_zedela_seq::MetodZedela::rankRMatrix() {
  std::vector<int> rMatrix(size * size + size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      rMatrix[i * (size + 1) + j] = matrix[i * size + j];
    }
    rMatrix[(i + 1) * (size + 1) - 1] = bVectrot[i];
  }
  return rankMatrix(rMatrix, size + 1);
}
