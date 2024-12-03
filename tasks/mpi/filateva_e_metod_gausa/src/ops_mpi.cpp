// Filateva Elizaveta Metod Gausa
#include "mpi/filateva_e_metod_gausa/include/ops_mpi.hpp"

#include <vector>
#include <limits>

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
    if (taskData->inputs_count[0] != taskData->outputs_count[0] || taskData->inputs_count[0] == 0) {
      return false;
    }
    size = taskData->inputs_count[0];

    auto* temp = reinterpret_cast<double*>(taskData->inputs[0]);
    this->matrix.insert(matrix.end(), temp, temp + size * size);

    temp = reinterpret_cast<double*>(taskData->inputs[1]);
    this->b_vector.insert(b_vector.end(), temp, temp + size);

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
        if (std::abs(temp_matrix[i * (size + 1) + j]) > epsilon){
          is_null_rows = false;
          is_null_rows_r = false; 
          break;
        }
        determenant *= temp_matrix[i * (size + 1) + i];
      }
      if (!is_null_rows) { rank_matrix++; }
      if (is_null_rows_r && std::abs(temp_matrix[i * (size + 1) + size]) > epsilon ) { is_null_rows_r = false; }
      if (!is_null_rows_r) { rank_r_matrix++; }
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

  boost::mpi::broadcast(world, size, 0);
  std::vector<double> t_strock(size, 0);
  int delta;
  int ost;

  std::vector<double> L;
  std::vector<double> U;
  std::vector<double> lU;
  std::vector<double> lL;

  if (world.rank() == 0) {
    L.resize(size * size, 0.0);
    L[0] = 1.0;
    U = matrix;
  } else {
    t_strock.resize(size, 0);
  }

  for (int i = 1; i < size; ++i) {
    if (world.size() > 1) {
      delta = (size - i) / (world.size() - 1);
      ost = (size - i) % (world.size() - 1);
    } else {
      delta = 0;
      ost = size - i;
    }

    if (world.rank() == 0) {
      if (delta != 0) {
        for (int proc = 0; proc < world.size() - 1; proc++) {
          world.send(proc + 1, 0, U.data() + proc * size * delta + (ost + i) * size, delta * size);
        }
      }
      lL.resize(ost, 0.0);
      lU.resize(ost * size, 0.0);
      std::copy(U.begin() + i * size, U.begin() + size * (ost + i), lU.begin());
      t_strock.assign(U.begin() + (i - 1) * size, U.begin() + i * size);
    } else {
      lL.resize(delta, 0.0);
      lU.resize(size * delta, 0.0);
      if (delta != 0) {
        world.recv(0, 0, lU.data(), delta * size);
      }
    }

    boost::mpi::broadcast(world, t_strock.data(), size, 0);

    for (long unsigned int j = 0; j < lL.size(); ++j) {
      lL[j] = lU[j * size + i - 1] / t_strock[i - 1];
      for (int k = i - 1; k < size; k++) {
        lU[j * size + k] -= lL[j] * t_strock[k];
      }
    }

    if (world.rank() == 0) {
      if (!lU.empty()) {
        std::copy(lU.begin(), lU.end(), U.begin() + i * size);
        for (long unsigned int j = 0; j < lL.size(); j++) {
          L[(i + j) * size + i - 1] = lL[j];
        }
      }

      std::vector<double> temp1(size * delta);
      std::vector<double> temp2(delta);
      if (delta != 0) {
        for (int proc = 0; proc < 2 * (world.size() - 1); proc++) {
          status = world.probe(boost::mpi::any_source, boost::mpi::any_tag);
          if (status.tag() == 0) {
            world.recv(status.source(), status.tag(), temp1.data(), delta * size);
            std::copy(temp1.begin(), temp1.end(), U.begin() + (status.source() - 1) * size * delta + (ost + i) * size);
          } else if (status.tag() == 1) {
            world.recv(status.source(), status.tag(), temp2.data(), delta);
            for (int j = 0; j < delta; j++) {
              L[(i + ost + (status.source() - 1) * delta + j) * size + i - 1] = temp2[j];
            }
          }
        }
      }
    } else {
      if (delta != 0) {
        world.send(0, 0, lU.data(), delta * size);
        world.send(0, 1, lL.data(), delta);
      }
    }

    if (world.rank() == 0) {
      L[i * size + i] = 1.0;
    }
  }

  if (world.rank() == 0) {
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
  }

  return true;
}

bool filateva_e_metod_gausa_mpi::MetodGausa::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(resh.data()));
  }
  return true;
}
