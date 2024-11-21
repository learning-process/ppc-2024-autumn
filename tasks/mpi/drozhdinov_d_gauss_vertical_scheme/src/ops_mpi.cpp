// Copyright 2023 Nesterov Alexander
#include "mpi/drozhdinov_d_gauss_vertical_scheme/include/ops_mpi.hpp"
// not example
#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>
#define GAMMA 1e-9

using namespace std::chrono_literals;

int makeLinCoords(int x, int y, int xSize) { return y * xSize + x; }

std::vector<double> drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel::GaussVeticalScheme(
    const std::vector<double>& matrix, int rows, int cols, const std::vector<double>& vec) {
  std::vector<double> b(rows);
  b = vec;
  const int delta = cols / world.size();
  const int r = cols % world.size();
  int proc_r = 0;
  std::vector<int> rs(world.size(), 0);
  if (world.rank() == 0) {
    for (int proc = 0; proc < world.size(); proc++) {
      rs[proc] = (proc < r ? 1 : 0);
    }
  }
  scatter(world, rs, proc_r, 0);
  // MPI_Scatter(rs, 1, MPI_INT, &proc_r, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<double> local_coefs((delta + proc_r) * rows);
  std::vector<double> current(rows);
  int* row_number = new int[rows];
  char* major = new char[rows];
  std::vector<double> result(rows);
  for (int row = 0; row < rows; row++) {
    row_number[row] = -1;
    major[row] = -1;
  }
  if (world.rank() == 0) {
    for (int col = 0; col < cols; col++) {
      if (col % world.size() != 0) {
        world.send(col % world.size(), 0, _coefs.data() + col * rows, rows);
      }
    }
    for (int row = 0; row < rows; row++) {
      for (int col_count = 0; col_count < delta + proc_r; col_count++) {
        local_coefs[makeLinCoords(row, col_count, cols)] = matrix[makeLinCoords(col_count * world.size(), row, cols)];
      }
    }
  } else {
    for (int row_count = 0; row_count < delta + proc_r; row_count++) {
      world.recv(0, 0, local_coefs.data() + row_count * rows, rows);
    }
  }
  world.barrier();

  int root, new_root;
  for (int curcol = 0; curcol < cols - 1; curcol++) {
    new_root = -1;
    root = -1;
    if (curcol % world.size() == world.rank()) {
      new_root = world.rank();
      double max = 0.0;
      int index = 0;
      for (int lcol = 0; lcol < rows; lcol++) {
        if ((abs(local_coefs[makeLinCoords(lcol, (curcol / world.size()), rows)]) >= abs(max)) &&
            (major[lcol] == -1)) {
          max = local_coefs[makeLinCoords(lcol, (curcol / world.size()), rows)];
          index = lcol;
        }
      }
      major[index] = 1;
      row_number[curcol] = index;
      for (int k = 0; k < rows; k++) {
        current[k] = 0.0;
        if (major[k] == -1) {
          current[k] = local_coefs[makeLinCoords(k, (curcol / world.size()), rows)] /
                       local_coefs[makeLinCoords(index, (curcol / world.size()), rows)];
        }
      }
    }
    all_reduce(world, new_root, root, boost::mpi::maximum<int>());
    broadcast(world, major, rows, root);  // ok
    broadcast(world, current.data(), rows, root);
    broadcast(world, row_number, rows, root);
    for (int lrow = 0; lrow < delta + proc_r; lrow++) {
      for (int lcol = 0; lcol < rows; lcol++) {
        if ((major[lcol] == -1) || (abs(local_coefs[makeLinCoords(lcol, lrow, rows)]) < GAMMA)) {
          local_coefs[makeLinCoords(lcol, lrow, rows)] -=
              current[lcol] * local_coefs[makeLinCoords(row_number[curcol], lrow, rows)];
          local_coefs[makeLinCoords(lcol, lrow, rows)] = (abs(local_coefs[makeLinCoords(lcol, lrow, rows)]) < GAMMA
                                                              ? 0
                                                              : local_coefs[makeLinCoords(lcol, lrow, rows)]);
        }
      }
    }
    if (world.rank() == 0) {
      for (int row = 0; row < rows; row++) {
        if (major[row] == -1) {
          b[row] -= b[row_number[curcol]] * current[row];
        }
      }
    }
  }
  for (int i = 0; i < rows; i++) {
    if (major[i] == -1) {
      row_number[rows - 1] = i;
      break;
    }
  }
  double d;
  if (world.rank() == 0) {
    std::vector<double> verh(cols * rows);
    for (int col = 0; col < cols; col++) {
      if (col % world.size() == 0) {
        for (int row = 0; row < rows; row++) {
          verh[col * rows + row] = local_coefs[(col / world.size()) * rows + row];
        }
      } else {
        world.recv(col % world.size(), 0, verh.data() + col * rows, rows);
      }
    }
    for (int n = rows - 1; n >= 0; n--) {
      d = 0.0;
      for (int k = n + 1; k < rows; k++) {
        d += result[k] * verh[k * cols + row_number[n]];
      }
      result[n] = (abs((b[row_number[n]] - d) / verh[n * cols + row_number[n]])) > GAMMA
                      ? (b[row_number[n]] - d) / verh[n * cols + row_number[n]]
                      : 0;
    }
  } else {
    for (int col_count = 0; col_count < delta + proc_r; col_count++) {
      world.send(0, 0, local_coefs.data() + col_count * rows, rows);
    }
  }
  return result;
}

std::vector<double> genElementaryMatrix(int rows, int columns) {
  std::vector<double> res;
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < columns; j++) {
      if (i == j) {
        res.push_back(1);
      } else {
        res.push_back(0);
      }
    }
  }
  return res;
}

std::vector<int> drozhdinov_d_gauss_vertical_scheme_mpi::getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 100;
  }
  return vec;
}

bool drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  coefs = std::vector<double>(taskData->inputs_count[0]);
  auto* ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
    coefs[i] = ptr[i];
  }
  b = std::vector<double>(taskData->inputs_count[1]);
  auto* ptr1 = reinterpret_cast<double*>(taskData->inputs[1]);
  for (unsigned int i = 0; i < taskData->inputs_count[1]; i++) {
    b[i] = ptr1[i];
  }
  columns = taskData->inputs_count[2];
  rows = taskData->inputs_count[3];
  return true;
}

bool drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return (taskData->inputs_count[3] == taskData->inputs_count[2] &&
          taskData->inputs_count[2] == taskData->outputs_count[0]);
}

bool drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  std::vector<double> result(rows);
  std::vector<double> current(rows);
  for (int i = 0; i < rows; i++) {
    major.push_back(0);
    row_number.push_back(0);
  }
  for (int i = 0; i < columns; i++) {  //????
    double max = 0;
    int index = 0;
    for (int j = 0; j < rows; j++) {
      if ((abs(coefs[makeLinCoords(j, i, columns)] >= abs(max)) && (!major[j]))) {
        max = coefs[makeLinCoords(j, i, columns)];
        index = j;
      }
    }
    major[index] = 1;
    row_number[i] = index;
    for (int ii = 0; ii < rows; ii++) {
      current[ii] = 0;
      if (!major[ii]) {
        current[ii] = coefs[makeLinCoords(i, ii, columns)] / coefs[makeLinCoords(i, index, columns)];
      }
    }
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < columns; column++) {
        if (!major[row]) {
          coefs[makeLinCoords(column, row, columns)] -= coefs[makeLinCoords(column, index, columns)] * current[row];
        }
      }
      if (!major[row]) {
        b[row] -= b[index] * current[row];
      }
    }
  }
  for (int k = 0; k < rows; k++) {
    if (!major[k]) {
      row_number[rows - 1] = k;
      break;
    }
  }
  for (int m = rows - 1; m >= 0; m--) {
    elem = 0;
    for (int n = m + 1; n < rows; n++) {
      elem += result[n] * coefs[makeLinCoords(n, row_number[m], columns)];
    }
    result[m] = (b[row_number[m]] - elem) / coefs[makeLinCoords(m, row_number[m], columns)];
  }
  for (auto v : result) {
    x.push_back(v);
  }
  return true;
}

bool drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  for (int i = 0; i < columns; i++) {
    reinterpret_cast<double*>(taskData->outputs[0])[i] = x[i];
  }
  return true;
}

bool drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    _rows = taskData->inputs_count[3];
    _columns = taskData->inputs_count[2];
  }
  broadcast(world, _columns, 0);
  broadcast(world, _rows, 0);
  // fbd nt ncssr dlt
  if (world.rank() == 0) {
    // Init vectors
    _coefs = std::vector<double>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    for (unsigned int i = 0; i < taskData->inputs_count[0]; i++) {
      _coefs[i] = tmp_ptr[i];
    }
    _b = std::vector<double>(taskData->inputs_count[1]);
    auto* ptr1 = reinterpret_cast<double*>(taskData->inputs[1]);
    for (unsigned int i = 0; i < taskData->inputs_count[1]; i++) {
      _b[i] = ptr1[i];
    }
  } else {
    _coefs = std::vector<double>(_columns * _rows);
    _b = std::vector<double>(_rows);
  }
  _x = std::vector<double>(_rows, 0);
  return true;
}

bool drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return (taskData->inputs_count[2] == taskData->inputs_count[3] &&
            taskData->inputs_count[2] == taskData->outputs_count[0]);
  }
  return true;
}

bool drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  _x = GaussVeticalScheme(_coefs, _rows, _columns, _b);
  return true;
}

bool drozhdinov_d_gauss_vertical_scheme_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (int i = 0; i < _columns; i++) {
      reinterpret_cast<double*>(taskData->outputs[0])[i] = _x[i];
    }
  }
  return true;
}