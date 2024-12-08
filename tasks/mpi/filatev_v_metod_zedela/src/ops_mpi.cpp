// Filatev Vladislav Metod Zedela
#include "mpi/filatev_v_metod_zedela/include/ops_mpi.hpp"

#include <vector>

filatev_v_metod_zedela_mpi::MetodZedela::MetodZedela(std::shared_ptr<ppc::core::TaskData> taskData_,
                                                     boost::mpi::communicator world_)
    : Task(std::move(taskData_)), world(std::move(world_)) {
  if (world.rank() == 0) {
    this->size = taskData->inputs_count[0];

    auto* temp = reinterpret_cast<int*>(taskData->inputs[0]);

    this->matrix.insert(matrix.end(), temp, temp + size * size);

    temp = reinterpret_cast<int*>(taskData->inputs[1]);
    this->bVectrot.insert(bVectrot.end(), temp, temp + size);
    this->tMatrix.resize(size * size);
    this->delit.resize(size);
  }
}

bool filatev_v_metod_zedela_mpi::MetodZedela::validation() {
  internal_order_test();
  if (world.size() == 1) {
    if (taskData->inputs_count[0] != taskData->outputs_count[0]) {
      return false;
    }
    int rank = rankMatrix(matrix, size);
    if (rank != rankRMatrix()) return false;
    if (rank == 0 || determinant() == 0) return false;
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

  boost::mpi::broadcast(world, size, 0);
  if (world.rank() != 0) {
    this->bVectrot.resize(size);
  }
  boost::mpi::broadcast(world, bVectrot.data(), size, 0);

  int mess;
  bool check = true;

  if (world.rank() == 0) {
    int rank = 0;
    int rankR = 0;
    bool st = true;
    int coll_proc = world.size() - 1;
    int task = 4;
    while (coll_proc != 0) {
      status = world.probe(boost::mpi::any_source, boost::mpi::any_tag);
      if (status.tag() == 0) {
        world.recv(status.source(), status.tag(), mess);
        if (task != 0) {
          world.send(status.source(), task, matrix);
          task--;
        } else {
          world.send(status.source(), 0, 1);
          coll_proc--;
        }
      } else {
        if (status.tag() == 1) {
          world.recv(status.source(), status.tag(), rank);
        }
        if (status.tag() == 2) {
          world.recv(status.source(), status.tag(), rankR);
        }
        if (status.tag() == 3) {
          bool temp;
          world.recv(status.source(), status.tag(), temp);
          st = st && temp;
        }
        if (status.tag() == 4) {
          bool temp;
          world.recv(status.source(), status.tag(), temp);
          st = st && temp;
        }
      }
    }
    check = (st && rank == rankR);
  } else {
    bool stop = true;
    while (stop) {
      world.send(0, 0, world.rank());
      status = world.probe(0, boost::mpi::any_tag);
      if (status.tag() == 0) {
        world.recv(status.source(), status.tag(), mess);
        stop = false;
      } else {
        matrix.resize(size * size);
        world.recv(status.source(), status.tag(), matrix);
        if (status.tag() == 1) {
          int rank = rankMatrix(matrix, size);
          world.send(0, 1, rank);
        }
        if (status.tag() == 2) {
          int rank = rankRMatrix();
          world.send(0, 2, rank);
        }
        if (status.tag() == 3) {
          bool st = (determinant() != 0);
          world.send(0, 3, st);
        }
        if (status.tag() == 4) {
          bool st = true;
          for (int i = 0; i < size; ++i) {
            int sum = 0;
            for (int j = 0; j < size; ++j) sum += abs(matrix[i * size + j]);
            sum -= abs(matrix[i * size + i]);
            if (sum > abs(matrix[i * size + i])) {
              st = false;
              break;
            }
          }
          world.send(0, 4, st);
        }
      }
    }
  }

  return check;
}

bool filatev_v_metod_zedela_mpi::MetodZedela::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        if (i == j) {
          delit[i] = matrix[i * size + j];
          tMatrix[i * size + j] = bVectrot[i];
        } else {
          tMatrix[i * size + j] = -matrix[i * size + j];
        }
      }
    }
  }
  return true;
}

bool filatev_v_metod_zedela_mpi::MetodZedela::run() {
  internal_order_test();
  std::vector<double> it1(size, 0);
  std::vector<double> it2(size);  // prev
  if (world.size() == 1 || size <= world.size()) {
    if (world.rank() == 0) {
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
            sum += it1[j] * tMatrix[i * size + j];
          }
          sum += tMatrix[(size + 1) * i];
          for (int j = i + 1; j < size; ++j) {
            sum += it2[j] * tMatrix[i * size + j];
          }
          it1[i] = sum / delit[i];
          sum1 += it1[i];
        }
      } while (abs(sum1 - sum2) > alfa);
      answer = it1;
    }
    return true;
  }

  int color = static_cast<int>(world.rank() != 0);
  boost::mpi::communicator new_comm = world.split(color);

  if (world.rank() != 0) {
    tMatrix.resize(size * size);
    delit.resize(size, 0);
  }
  boost::mpi::broadcast(world, tMatrix.data(), size * size, 0);
  boost::mpi::broadcast(world, delit.data(), size, 0);

  int delta = size / (world.size() - 1);
  int ost = size % (world.size() - 1);

  if (world.rank() == 0) {
    double max_z = 0;
    do {
      max_z = 0;
      std::swap(it1, it2);
      world.send(1, 0, it2);
      world.recv(1, 0, it1);

      for (int i = 2; i < world.size(); i++) {
        world.send(i, 0, it1);
        world.recv(i, 0, it1);
      }
      for (int i = 0; i < size; i++) {
        max_z += abs(it1[i] - it2[i]);
      }

      if (max_z > alfa) {
        for (int i = 1; i < world.size(); i++) {
          world.send(i, 0, false);
        }
      } else {
        for (int i = 1; i < world.size(); i++) {
          world.send(i, 0, true);
        }
        break;
      }

    } while (true);
    answer = it1;

  } else {
    bool stop = false;
    do {
      std::vector<double> temp(size, 0);
      if (world.rank() == 1) {
        world.recv(0, 0, it2);
      }
      boost::mpi::broadcast(new_comm, it2.data(), size, 0);
      int start = world.rank() <= ost ? (world.rank() - 1) * (delta + 1)
                                      : ((delta + 1) * ost + (world.rank() - ost - 1) * delta);
      int end = start + (world.rank() <= ost ? (delta + 1) : delta);
      for (int i = start; i < end; i++) {
        temp[i] = tMatrix[(size + 1) * i];
        for (int j = i + 1; j < size; ++j) {
          temp[i] += it2[j] * tMatrix[i * size + j];
        }
      }
      if (world.rank() != 1) {
        world.recv(0, 0, it1);
      }
      for (int i = start; i < end; i++) {
        for (int j = 0; j < i; ++j) {
          temp[i] += it1[j] * tMatrix[i * size + j];
        }
        it1[i] = temp[i] / delit[i];
      }
      world.send(0, 0, it1);
      world.recv(0, 0, stop);
    } while (!stop);
  }

  return true;
}

bool filatev_v_metod_zedela_mpi::MetodZedela::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(answer.data()));
  }
  return true;
}

void filatev_v_metod_zedela_mpi::MetodZedela::setAlfa(double _alfa) { this->alfa = _alfa / 100; }

int filatev_v_metod_zedela_mpi::MetodZedela::rankMatrix(std::vector<int>& matrixT, int n) const {
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

double filatev_v_metod_zedela_mpi::MetodZedela::determinant() {
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

int filatev_v_metod_zedela_mpi::MetodZedela::rankRMatrix() {
  std::vector<int> rMatrix(size * size + size);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      rMatrix[i * (size + 1) + j] = matrix[i * size + j];
    }
    rMatrix[(i + 1) * (size + 1) - 1] = bVectrot[i];
  }
  return rankMatrix(rMatrix, size + 1);
}