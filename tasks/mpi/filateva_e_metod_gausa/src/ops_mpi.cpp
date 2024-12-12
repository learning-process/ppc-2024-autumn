// Filateva Elizaveta Metod Gausa
#include "mpi/filateva_e_metod_gausa/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <limits>
#include <vector>

bool filateva_e_metod_gausa_mpi::MetodGausa::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    resh.resize(size, 0);
  }
  return true;
}

bool filateva_e_metod_gausa_mpi::MetodGausa::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs_count[0] <= 0 || taskData->inputs_count[0] != taskData->outputs_count[0]) {
      return false;
    }
    size = taskData->inputs_count[0];

    auto* temp = reinterpret_cast<double*>(taskData->inputs[0]);
    this->matrix.insert(matrix.end(), temp, temp + size * size);

    temp = reinterpret_cast<double*>(taskData->inputs[1]);
    this->b_vector.insert(b_vector.end(), temp, temp + size);

    std::vector<double> temp_line(size);
    for (int i = 0; i < size; i++) {
      if (matrix[i * size + i] == 0) {
        bool found = false;
        for (int j = 0; j < size; j++) {
          if (j != i && matrix[j * size + i] != 0) {
            std::copy(matrix.begin() + i * size, matrix.begin() + (i + 1) * size, temp_line.begin());
            std::copy(matrix.begin() + j * size, matrix.begin() + (j + 1) * size, matrix.begin() + i * size);
            std::copy(temp_line.begin(), temp_line.end(), matrix.begin() + j * size);
            std::swap(b_vector[i], b_vector[j]);
            found = true;
            break;
          }
        }
        if (!found) {
          return false;
        }
      }
    }

    std::vector<double> temp_matrix(size * (size + 1));
    for (int i = 0; i < size; i++) {
      std::copy(matrix.begin() + i * size, matrix.begin() + (i + 1) * size, temp_matrix.begin() + i * (size + 1));
      temp_matrix[i * (size + 1) + size] = b_vector[i];
    }

    for (int r = 0; r < size; r++) {
      for (int j = r + 1; j < size; j++) {
        double factor = temp_matrix[j * (size + 1) + r] / temp_matrix[r * (size + 1) + r];
        for (int k = r; k < size + 1; k++) {
          temp_matrix[j * (size + 1) + k] -= factor * temp_matrix[r * (size + 1) + k];
        }
      }
    }

    int rank_matrix = size;
    int rank_r_matrix = size;
    double determenant = 1;

    double epsilon = std::numeric_limits<double>::epsilon();

    for (int i = 0; i < size; i++) {
      bool is_null_rows = true;
      bool is_null_rows_r = true;
      for (int j = 0; j < size; j++) {
        if (std::abs(temp_matrix[i * (size + 1) + j]) > epsilon) {
          is_null_rows = false;
          is_null_rows_r = false;
          break;
        }
        determenant *= temp_matrix[i * (size + 1) + i];
      }
      if (!is_null_rows) {
        rank_matrix++;
      }
      if (is_null_rows_r && std::abs(temp_matrix[i * (size + 1) + size]) > epsilon) {
        is_null_rows_r = false;
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

  std::vector<double> temp_matrix;
  boost::mpi::broadcast(world, size, 0);
  int size_n = size + 1;

  if (world.rank() == 0) {
    temp_matrix.resize(size * size_n);
    for (int i = 0; i < size; i++) {
      std::copy(matrix.begin() + i * size, matrix.begin() + (i + 1) * size, temp_matrix.begin() + i * size_n);
      temp_matrix[i * (size + 1) + size] = b_vector[i];
    }
  }

  if (world.size() == 1) {
    for (int r = 0; r < size; r++) {
      for (int j = r + 1; j < size; j++) {
        double factor = temp_matrix[j * (size + 1) + r] / temp_matrix[r * (size + 1) + r];
        for (int k = r; k < size + 1; k++) {
          temp_matrix[j * (size + 1) + k] -= factor * temp_matrix[r * (size + 1) + k];
        }
      }
    }
    for (int i = size - 1; i >= 0; i--) {
      resh[i] = temp_matrix[(i + 1) * (size + 1) - 1];
      for (int j = i + 1; j < size; j++) {
        resh[i] -= temp_matrix[i * (size + 1) + j] * resh[j];
      }
      resh[i] /= temp_matrix[i * (size + 1) + i];
    }
    return true;
  }

  int delta = size / (world.size() - 1);
  int ost = size % (world.size() - 1);
  std::vector<double> local_matrix(size * size_n, 0);
  std::vector<double> t_strock(size_n, 0);
  std::vector<int> distribution(world.size(), 0);
  std::vector<int> displacement(world.size(), 0);

  for (int i = 1; i < size; ++i) {
    int size_m = 0;
    delta = (size - i) / (world.size() - 1);
    ost = (size - i) % (world.size() - 1);

    distribution.assign(world.size(), delta * size_n);
    distribution[0] = ost * size_n;
    int n = -2;
    std::generate(displacement.begin(), displacement.end(), [&]() { return (++n * delta + ost) * size_n; });
    displacement[0] = 0;

    size_m = world.rank() != 0 ? delta : ost;

    boost::mpi::scatterv(world, temp_matrix.data() + i * size_n, distribution, displacement, local_matrix.data(),
                         size_m * size_n, 0);

    if (world.rank() == 0) {
      std::copy(temp_matrix.begin() + (i - 1) * size_n, temp_matrix.begin() + i * size_n, t_strock.begin());
    }
    boost::mpi::broadcast(world, t_strock, 0);

    for (int j = 0; j < size_m; j++) {
      double factor = local_matrix[j * size_n + i - 1] / t_strock[i - 1];
      local_matrix[j * size_n + i - 1] = 0;
      for (int k = i; k < size_n; k++) {
        local_matrix[j * size_n + k] -= factor * t_strock[k];
      }
    }

    boost::mpi::gatherv(world, local_matrix.data(), size_m * size_n, temp_matrix.data() + i * size_n, distribution,
                        displacement, 0);
  }

  if (world.rank() == 0) {
    for (int i = size - 1; i >= 0; i--) {
      resh[i] = temp_matrix[(i + 1) * (size + 1) - 1];
      for (int j = i + 1; j < size; j++) {
        resh[i] -= temp_matrix[i * (size + 1) + j] * resh[j];
      }
      resh[i] /= temp_matrix[i * (size + 1) + i];
    }
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
