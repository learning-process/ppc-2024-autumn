// Filateva Elizaveta Metod Gausa
#include "mpi/filateva_e_metod_gausa/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <limits>
#include <vector>

bool filateva_e_metod_gausa_mpi::MetodGausa::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    size = taskData->inputs_count[0];
    auto* temp = reinterpret_cast<double*>(taskData->inputs[0]);
    this->matrix.assign(temp, temp + size * size);
    temp = reinterpret_cast<double*>(taskData->inputs[1]);
    this->vecB.assign(temp, temp + size);
    resh.resize(size, 0);
  }
  return true;
}

bool filateva_e_metod_gausa_mpi::MetodGausa::validation() {
  internal_order_test();
  if (world.rank() == 0) {
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
        bool found = false;
        for (int j = i; j < size; j++) {
          if (rMatrix[j].pLine[i] != 0) {
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
  }
  return true;
}

bool filateva_e_metod_gausa_mpi::MetodGausa::run() {
  internal_order_test();

  std::vector<Matrix> rMatrix;
  if (world.rank() == 0) {
    for (int i = 0; i < size; ++i) {
      rMatrix.push_back({&matrix[i * size], vecB[i]});
    }
  }

  if (world.size() == 1) {
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

  boost::mpi::broadcast(world, size, 0);
  int size_n = size + 1;
  int delta = size / world.size();
  int ost = size % world.size();
  int size_m = world.rank() < ost ? delta + 1 : delta;

  std::vector<double> tem_matrix;
  std::vector<double> local_matrix(size_n * size_m);
  std::vector<double> temp(size_n);
  std::vector<int> rows(size_m);
  resh.resize(size, 0);

  for (int i = 0; i < size_m; i++) {
    rows[i] = world.rank() + world.size() * i;
  }

  if (world.rank() == 0) {
    for (int i = 0; i < size; i++) {
      if (rMatrix[i].pLine[i] == 0) {
        for (int j = 0; j < size; j++) {
          if (j > i && rMatrix[j].pLine[i] != 0) {
            std::swap(rMatrix[i], rMatrix[j]);
            break;
          }
          if (j < i && rMatrix[j].pLine[i] != 0 && rMatrix[i].pLine[j] != 0) {
            std::swap(rMatrix[i], rMatrix[j]);
            break;
          }
        }
      }
    }
    tem_matrix.resize(size * size_n);
    for (int i = 0; i < size; i++) {
      std::copy(rMatrix[i].pLine, rMatrix[i].pLine + size, tem_matrix.begin() + i * size_n);
      tem_matrix[i * size_n + size] = rMatrix[i].b;
    }
  }

  std::vector<int> distribution(ost, (delta + 1) * size_n);
  distribution.insert(distribution.end(), world.size() - ost, delta * size_n);
  std::vector<int> displacement(world.size(), 0);
  for (int i = 1; i < world.size(); i++) {
    displacement[i] = displacement[i - 1] + ((i <= ost) ? (delta + 1) : delta) * size_n;
  }
  boost::mpi::scatterv(world, tem_matrix.data(), distribution, displacement, local_matrix.data(), size_m * size_n, 0);

  double epsilon = std::numeric_limits<double>::epsilon();

  int row = 0;
  for (int i = 0; i < size - 1;) {
    if (row < (int)rows.size() && i == rows[row]) {
      for (int j = 0; j < size_n; j++) {
        temp[j] = local_matrix[row * size_n + j];
      }
      boost::mpi::broadcast(world, temp, world.rank());
      if (std::abs(local_matrix[row * size_n + i]) < epsilon) {
        tem_matrix.resize(size * size_n);
        boost::mpi::gatherv(world, local_matrix.data(), size_m * size_n, tem_matrix.data(), distribution, displacement,
                            world.rank());
        for (int j = i + 1; j < size; j++) {
          if (tem_matrix[j * size_n + i] != 0) {
            std::copy(tem_matrix.data() + i * size_n, tem_matrix.data() + (i + 1) * size_n, temp.data());
            std::copy(tem_matrix.data() + j * size_n, tem_matrix.data() + (j + 1) * size_n,
                      tem_matrix.data() + i * size_n);
            std::copy(temp.begin(), temp.end(), tem_matrix.data() + j * size_n);
            break;
          }
        }
        boost::mpi::scatterv(world, tem_matrix.data(), distribution, displacement, local_matrix.data(), size_m * size_n,
                             world.rank());
        continue;
      }

      row++;
    } else {
      boost::mpi::broadcast(world, temp, i % world.size());
      if (std::abs(temp[i]) < epsilon) {
        boost::mpi::gatherv(world, local_matrix.data(), size_m * size_n, tem_matrix.data(), distribution, displacement,
                            i % world.size());
        boost::mpi::scatterv(world, tem_matrix.data(), distribution, displacement, local_matrix.data(), size_m * size_n,
                             i % world.size());
        continue;
      }
    }

    for (int j = row; j < size_m; j++) {
      double coeff = local_matrix[j * size_n + i] / temp[i];
      for (int k = i; k < size_n; k++) {
        local_matrix[j * size_n + k] -= coeff * temp[k];
      }
    }
    i++;
  }

  row = 0;
  for (auto i : rows) {
    resh[i] = local_matrix[row * size_n + size];
    row++;
  }

  row = size_m - 1;
  for (int i = size - 1; i > 0; i--) {
    if (row >= 0) {
      if (!rows.empty() && rows[row] == i) {
        resh[i] /= local_matrix[row * size_n + i];
        boost::mpi::broadcast(world, resh[i], world.rank());
        row--;
      } else {
        boost::mpi::broadcast(world, resh[i], i % world.size());
      }
    } else {
      boost::mpi::broadcast(world, resh[i], i % world.size());
    }
    for (int j = 0; j <= row; j++) {
      resh[rows[j]] -= local_matrix[j * size_n + i] * resh[i];
    }
  }

  if (world.rank() == 0) {
    resh[0] /= local_matrix[row * size_n];
  }

  return true;
}

bool filateva_e_metod_gausa_mpi::MetodGausa::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* output_data = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(resh.begin(), resh.end(), output_data);
  }
  return true;
}
