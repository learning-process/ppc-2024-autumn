// Filateva Elizaveta Metod Gausa

#include "seq/filateva_e_metod_gausa/include/ops_seq.hpp"

#include <iostream>
#include <limits>
#include <string>

bool filateva_e_metod_gausa_seq::MetodGausa::pre_processing() {
  internal_order_test();

  resh.resize(size, 0);
  auto* temp = reinterpret_cast<double*>(taskData->inputs[0]);
  this->matrix.assign(temp, temp + size * size);
  temp = reinterpret_cast<double*>(taskData->inputs[1]);
  this->vecB.assign(temp, temp + size);

  return true;
}

bool filateva_e_metod_gausa_seq::MetodGausa::validation() {
  internal_order_test();
  if (taskData->inputs_count[0] != taskData->outputs_count[0] || taskData->inputs_count[0] == 0) {
    return false;
  }
  size = taskData->inputs_count[0];
  std::vector<double> local_matrix;
  std::vector<double> local_vecB;
  std::vector<Matrix> rMatrix(size);

  auto* temp = reinterpret_cast<double*>(taskData->inputs[0]);
  local_matrix.assign(temp, temp + size * size);
  temp = reinterpret_cast<double*>(taskData->inputs[1]);
  local_vecB.assign(temp, temp + size);

  for (int i = 0; i < size; ++i) {
    rMatrix[i] = {&local_matrix[i * size], local_vecB[i]};
  }

  for (int i = 0; i < size; i++) {
    if (rMatrix[i].pLine[i] == 0) {
      bool found = false;
      for (int j = 0; j < size; j++) {
        if (j > i && rMatrix[j].pLine[i] != 0) {
          std::swap(rMatrix[i], rMatrix[j]);
          found = true;
          break;
        }
        if (j < i && rMatrix[j].pLine[i] != 0 && rMatrix[i].pLine[j] != 0) {
          std::swap(rMatrix[i], rMatrix[j]);
          found = true;
          break;
        }
      }
      if (!found) {
        break;
        return false;
      }
    }
  }

  for (int i = 0; i < size; i++) {
    if (rMatrix[i].pLine[i] == 0) {
      for (int j = i; j < size; j++) {
        if (rMatrix[j].pLine[i] != 0) {
          std::swap(rMatrix[i], rMatrix[j]);
          break;
        }
      }
    }
    for (int k = i + 1; k < size; k++) {
      double coeff = rMatrix[k].pLine[i] / rMatrix[i].pLine[i];
      for (int j = i; j < size; j++) {
        rMatrix[k].pLine[j] -= coeff * rMatrix[i].pLine[j];
      }
      rMatrix[k].b -= coeff * rMatrix[i].b;
    }
  }

  int rank_matrix = 0;
  int rank_r_matrix = 0;
  double determenant = 1;

  double epsilon = std::numeric_limits<double>::epsilon();

  for (int i = 0; i < size; i++) {
    bool is_null_rows = true;
    bool is_null_rows_r = true;
    for (int j = 0; j < size; j++) {
      if (std::abs(rMatrix[i].pLine[j]) > epsilon) {
        is_null_rows = false;
        is_null_rows_r = false;
        break;
      }
      is_null_rows_r = is_null_rows_r && std::abs(rMatrix[i].b) <= epsilon;
      determenant *= rMatrix[i].pLine[i];
    }
    if (!is_null_rows) {
      rank_matrix++;
    }
    if (!is_null_rows_r) {
      rank_r_matrix++;
    }
  }

  if (rank_matrix != rank_r_matrix) {
    return false;
  }
  if (std::abs(determenant) < epsilon) {
    return false;
  }

  return true;
}

bool filateva_e_metod_gausa_seq::MetodGausa::run() {
  internal_order_test();
  std::vector<Matrix> rMatrix(size);

  for (int i = 0; i < size; ++i) {
    rMatrix[i] = {&matrix[i * size], vecB[i]};
  }

  for (int i = 0; i < size; i++) {
    if (rMatrix[i].pLine[i] == 0) {
      for (int j = i; j < size; j++) {
        if (rMatrix[j].pLine[i] != 0) {
          std::swap(rMatrix[i], rMatrix[j]);
          break;
        }
      }
    }
    for (int k = i + 1; k < size; k++) {
      double coeff = rMatrix[k].pLine[i] / rMatrix[i].pLine[i];
      for (int j = i; j < size; j++) {
        rMatrix[k].pLine[j] -= coeff * rMatrix[i].pLine[j];
      }
      rMatrix[k].b -= coeff * rMatrix[i].b;
    }
  }

  for (int i = size - 1; i >= 0; i--) {
    resh[i] = rMatrix[i].b;
    for (int j = i + 1; j < size; j++) {
      resh[i] -= rMatrix[i].pLine[j] * resh[j];
    }
    resh[i] /= rMatrix[i].pLine[i];
  }

  return true;
}

bool filateva_e_metod_gausa_seq::MetodGausa::post_processing() {
  internal_order_test();
  auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(resh.begin(), resh.end(), output_data);
  return true;
}
