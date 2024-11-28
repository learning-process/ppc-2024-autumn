// Filateva Elizaveta Metod Gausa

#include "seq/filateva_e_metod_gausa/include/ops_seq.hpp"

#include <thread>

bool filateva_e_metod_gausa_seq::MetodGausa::pre_processing() {
  internal_order_test();
  
  size = taskData->inputs_count[0];
  auto* temp = reinterpret_cast<double*>(taskData->inputs[0]);
  this->matrix.insert(matrix.end(), temp, temp + size * size);

  temp = reinterpret_cast<double*>(taskData->inputs[1]);
  this->b_vector.insert(b_vector.end(), temp, temp + size);

  resh.resize(size,0);

  return true;
}

bool filateva_e_metod_gausa_seq::MetodGausa::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->outputs_count[0];
}

bool filateva_e_metod_gausa_seq::MetodGausa::run() {
  internal_order_test();
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

  std::vector<double> y(size);
  for (int i = 0; i < size; i++) {
      y[i] = b_vector[i];
      for (int j = 0; j < i; j++) {
        y[i] -= L[i * size + j] * y[j];
      }
  }

  for (int i = size - 1; i >= 0; i--) {
    resh[i] = y[i];
    for (int j = i + 1; j < size; j++) {
      resh[i] -= U[i * size + j] * resh[j];
    }
    resh[i] /= U[i * size + i];
  }

  return true;
}

bool filateva_e_metod_gausa_seq::MetodGausa::post_processing() {
  internal_order_test();
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(resh.data()));
  return true;
}
