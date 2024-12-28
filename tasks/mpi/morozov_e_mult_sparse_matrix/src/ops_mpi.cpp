#include "mpi/morozov_e_mult_sparse_matrix/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>
#define vec_double std::vector<double>
using namespace std::chrono_literals;
// for usability debugging
template <typename T>
void morozov_e_mult_sparse_matrix::printMatrix(std::vector<std::vector<T>> m) {
  for (int i = 0; i < m.size(); ++i) {
    for (int j = 0; j < m[0].size(); ++j) {
      std::cout << m[i][j] << " ";
    }
    std::cout << "\n";
  }
}
// for usability debugging
template <typename T>
void morozov_e_mult_sparse_matrix::printVector(std::vector<T> v) {
  for (int i = 0; i < v.size(); ++i) {
    std::cout << v[i] << " ";
  }
  std::cout << "\n";
}
void morozov_e_mult_sparse_matrix::convertToCCS(const std::vector<std::vector<double>>& matrix,
                                                std::vector<double>& values, std::vector<int>& row_indices,
                                                std::vector<int>& col_pointers) {
  int num_rows = matrix.size();
  int num_cols = num_rows > 0 ? matrix[0].size() : 0;

  col_pointers.resize(num_cols + 1, 0);

  for (int col = 0; col < num_cols; ++col) {
    for (int row = 0; row < num_rows; ++row) {
      if (matrix[row][col] != 0) {
        col_pointers[col + 1]++;
      }
    }
  }
  for (int col = 0; col < num_cols; ++col) {
    col_pointers[col + 1] += col_pointers[col];
  }
  for (int col = 0; col < num_cols; ++col) {
    for (int row = 0; row < num_rows; ++row) {
      double value = matrix[row][col];
      if (value != 0) {
        values.push_back(value);
        row_indices.push_back(row);
        col_pointers[col]++;
      }
    }
  }

  for (int col = num_cols; col > 0; --col) {
    col_pointers[col] = col_pointers[col - 1];
  }
  col_pointers[0] = 0;
}
template <typename T>
T morozov_e_mult_sparse_matrix::scalMultOfVectors(const std::vector<T>& vA, const std::vector<T>& vB) {
  double ans = 0;
  for (size_t i = 0; i < vA.size(); ++i) {
    ans += vA[i] * vB[i];
  }
  return ans;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
morozov_e_mult_sparse_matrix::convertToBasicMatrixs(const std::vector<double>& dA, const std::vector<int>& row_indA,
                                                    const std::vector<int>& col_indA, const std::vector<double>& dB,
                                                    const std::vector<int>& row_indB, const std::vector<int>& col_indB,
                                                    int rowsA, int columnsA, int rowsB, int columnsB) {
  std::vector<std::vector<double>> matrixA(rowsA, std::vector<double>(columnsA, 0));
  for (int i = 0; i < rowsA; ++i) {
    std::vector<double> v(col_indA.size() - 1, 0);
    for (size_t j = 0; j < col_indA.size() - 1; ++j) {
      for (int ind = col_indA[j]; ind < col_indA[j + 1]; ++ind) {
        if (row_indA[ind] == i) {
          v[j] = dA[ind];
        }
      }
    }
    matrixA[i] = v;
  }
  std::vector<std::vector<double>> matrixB(columnsB, std::vector<double>(rowsB, 0));
  for (size_t i = 0; i < col_indB.size() - 1; ++i) {
    std::vector<double> v(rowsB, 0);
    for (int ind = col_indB[i]; ind < col_indB[i + 1]; ++ind) {
      v[row_indB[ind]] = dB[ind];
    }
    matrixB[i] = v;
  }

  return std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>(matrixA, matrixB);
}
bool morozov_e_mult_sparse_matrix::TestTaskSequential::pre_processing() {
  internal_order_test();
  rowsA = taskData->inputs_count[0];
  columnsA = taskData->inputs_count[1];
  dA_size = taskData->inputs_count[2];

  row_indA_size = taskData->inputs_count[3];
  col_indA_size = taskData->inputs_count[4];
  rowsB = taskData->inputs_count[5];
  columnsB = taskData->inputs_count[6];
  dB_size = taskData->inputs_count[7];
  row_indB_size = taskData->inputs_count[8];
  col_indB_size = taskData->inputs_count[9];
  dA.resize(dA_size);
  for (int i = 0; i < dA_size; ++i) {
    auto* dA_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    dA[i] = dA_ptr[i];
  }
  row_indA.resize(row_indA_size);
  for (int i = 0; i < row_indA_size; ++i) {
    int* row_indA_ptr = reinterpret_cast<int*>(taskData->inputs[1]);
    row_indA[i] = row_indA_ptr[i];
  }
  col_indA.resize(col_indA_size);
  for (int i = 0; i < col_indA_size; ++i) {
    int* col_indA_ptr = reinterpret_cast<int*>(taskData->inputs[2]);
    col_indA[i] = col_indA_ptr[i];
  }

  dB.resize(dB_size);
  for (int i = 0; i < dB_size; ++i) {
    auto* dB_ptr = reinterpret_cast<double*>(taskData->inputs[3]);
    dB[i] = dB_ptr[i];
  }
  row_indB.resize(row_indB_size);
  for (int i = 0; i < row_indB_size; ++i) {
    int* row_indB_ptr = reinterpret_cast<int*>(taskData->inputs[4]);
    row_indB[i] = row_indB_ptr[i];
  }
  col_indB.resize(col_indB_size);
  for (int i = 0; i < col_indB_size; ++i) {
    int* col_indB_ptr = reinterpret_cast<int*>(taskData->inputs[5]);
    col_indB[i] = col_indB_ptr[i];
  }

  return true;
}

bool morozov_e_mult_sparse_matrix::TestTaskSequential::validation() {
  internal_order_test();
  int rA = taskData->inputs_count[0];
  int cA = taskData->inputs_count[1];
  int rB = taskData->inputs_count[5];
  int cB = taskData->inputs_count[6];
  int rowsAns = taskData->outputs_count[0];
  int columnsAns = taskData->outputs_count[1];
  return cA == rB && columnsAns == cB && rowsAns == rA;
}

bool morozov_e_mult_sparse_matrix::TestTaskSequential::run() {
  internal_order_test();
  auto pairMatrix = morozov_e_mult_sparse_matrix::convertToBasicMatrixs(dA, row_indA, col_indA, dB, row_indB, col_indB,
                                                                        rowsA, columnsA, rowsB, columnsB);
  ans.resize(rowsA, std::vector<double>(columnsB, 0));
  for (int i = 0; i < rowsA; ++i) {
    for (int j = 0; j < columnsB; ++j) {
      ans[i][j] = morozov_e_mult_sparse_matrix::scalMultOfVectors(pairMatrix.first[i], pairMatrix.second[j]);
    }
  }
  return true;
}

bool morozov_e_mult_sparse_matrix::TestTaskSequential::post_processing() {
  internal_order_test();
  for (size_t i = 0; i < ans.size(); ++i) {
    for (size_t j = 0; j < ans[i].size(); ++j) {
      reinterpret_cast<double*>(taskData->outputs[i])[j] = ans[i][j];
    }
  }
  return true;
}

bool morozov_e_mult_sparse_matrix::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    rowsA = taskData->inputs_count[0];
    columnsA = taskData->inputs_count[1];
    dA_size = taskData->inputs_count[2];

    row_indA_size = taskData->inputs_count[3];
    col_indA_size = taskData->inputs_count[4];
    rowsB = taskData->inputs_count[5];
    columnsB = taskData->inputs_count[6];
    dB_size = taskData->inputs_count[7];
    row_indB_size = taskData->inputs_count[8];
    col_indB_size = taskData->inputs_count[9];
    dA.resize(dA_size);
    for (int i = 0; i < dA_size; ++i) {
      auto* dA_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
      dA[i] = dA_ptr[i];
    }
    row_indA.resize(row_indA_size);
    for (int i = 0; i < row_indA_size; ++i) {
      int* row_indA_ptr = reinterpret_cast<int*>(taskData->inputs[1]);
      row_indA[i] = row_indA_ptr[i];
    }
    col_indA.resize(col_indA_size);
    for (int i = 0; i < col_indA_size; ++i) {
      int* col_indA_ptr = reinterpret_cast<int*>(taskData->inputs[2]);
      col_indA[i] = col_indA_ptr[i];
    }

    dB.resize(dB_size);
    for (int i = 0; i < dB_size; ++i) {
      auto* dB_ptr = reinterpret_cast<double*>(taskData->inputs[3]);
      dB[i] = dB_ptr[i];
    }
    row_indB.resize(row_indB_size);
    for (int i = 0; i < row_indB_size; ++i) {
      int* row_indB_ptr = reinterpret_cast<int*>(taskData->inputs[4]);
      row_indB[i] = row_indB_ptr[i];
    }
    col_indB.resize(col_indB_size);
    for (int i = 0; i < col_indB_size; ++i) {
      int* col_indB_ptr = reinterpret_cast<int*>(taskData->inputs[5]);
      col_indB[i] = col_indB_ptr[i];
    }
  }
  return true;
}

bool morozov_e_mult_sparse_matrix::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    int rA = taskData->inputs_count[0];
    int cA = taskData->inputs_count[1];
    int rB = taskData->inputs_count[5];
    int cB = taskData->inputs_count[6];
    int rowsAns = taskData->outputs_count[0];
    int columnsAns = taskData->outputs_count[1];
    return cA == rB && columnsAns == cB && rowsAns == rA;
  }
  return true;
}

bool morozov_e_mult_sparse_matrix::TestMPITaskParallel::run() {
  internal_order_test();
  int sizeA;
  int sizeB;
  int countVectorsForMult;
  if (world.rank() == 0) {
    sizeA = columnsA;
    sizeB = rowsB;
    countVectorsForMult = rowsA * columnsB;
  }
  broadcast(world, sizeA, 0);
  broadcast(world, sizeB, 0);
  broadcast(world, countVectorsForMult, 0);
  std::vector<std::pair<vec_double, vec_double>> vectorFor0Proc;

  if (world.rank() == 0) {
    auto pairMatrix = morozov_e_mult_sparse_matrix::convertToBasicMatrixs(dA, row_indA, col_indA, dB, row_indB,
                                                                          col_indB, rowsA, columnsA, rowsB, columnsB);
    ans.resize(rowsA, std::vector<double>(columnsB, 0));
    int cur_status_vector = 0;
    for (int i = 0; i < rowsA; ++i) {
      for (int j = 0; j < columnsB; ++j) {
        if (cur_status_vector % world.size() != 0) {
          world.send(cur_status_vector % world.size(), 0, &i, 1);
          world.send(cur_status_vector % world.size(), 0, &j, 1);
          world.send(cur_status_vector % world.size(), 0, pairMatrix.first[i].data(), sizeA);
          world.send(cur_status_vector % world.size(), 0, pairMatrix.second[j].data(), sizeB);
        }
        cur_status_vector++;
      }
    }
    cur_status_vector = 0;
    for (int i = 0; i < rowsA; ++i) {
      for (int j = 0; j < columnsB; ++j) {
        if (cur_status_vector % world.size() != 0) {
          int posA;
          int posB;
          double value;
          world.recv(cur_status_vector % world.size(), 0, &posA, 1);
          world.recv(cur_status_vector % world.size(), 0, &posB, 1);
          world.recv(cur_status_vector % world.size(), 0, &value, 1);
          // std::cout << "Answer from proc" << posA << posB << value << "\n";
          ans[posA][posB] = value;
        } else {
          ans[i][j] = morozov_e_mult_sparse_matrix::scalMultOfVectors(pairMatrix.first[i], pairMatrix.second[j]);
        }
        cur_status_vector++;
      }
    }
    // std::cout << world.rank() << "\n";
    // morozov_e_mult_sparse_matrix::printMatrix(ans);
  } else {
    for (int i = 0; i < countVectorsForMult / world.size(); ++i) {
      int posA;
      int posB;
      local_input_A = std::vector<double>(sizeA);
      local_input_B = std::vector<double>(sizeA);
      world.recv(0, 0, &posA, 1);
      world.recv(0, 0, &posB, 1);
      world.recv(0, 0, local_input_A.data(), sizeA);
      world.recv(0, 0, local_input_B.data(), sizeA);
      double value = morozov_e_mult_sparse_matrix::scalMultOfVectors(local_input_A, local_input_B);
      world.send(0, 0, &posA, 1);
      world.send(0, 0, &posB, 1);
      world.send(0, 0, &value, 1);
    }
    if (world.rank() < countVectorsForMult % world.size()) {
      int posA;
      int posB;
      local_input_A = std::vector<double>(sizeA);
      local_input_B = std::vector<double>(sizeA);
      world.recv(0, 0, &posA, 1);
      world.recv(0, 0, &posB, 1);
      world.recv(0, 0, local_input_A.data(), sizeA);
      world.recv(0, 0, local_input_B.data(), sizeA);
      double value = morozov_e_mult_sparse_matrix::scalMultOfVectors(local_input_A, local_input_B);
      world.send(0, 0, &posA, 1);
      world.send(0, 0, &posB, 1);
      world.send(0, 0, &value, 1);
    }
  }

  return true;
}

bool morozov_e_mult_sparse_matrix::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    for (size_t i = 0; i < ans.size(); ++i) {
      for (size_t j = 0; j < ans[i].size(); ++j) {
        reinterpret_cast<double*>(taskData->outputs[i])[j] = ans[i][j];
      }
    }
  }
  return true;
}
void morozov_e_mult_sparse_matrix::fillData(std::shared_ptr<ppc::core::TaskData>& taskData, int rowsA, int columnsA,
                                            int rowsB, int columnsB, std::vector<double>& dA,
                                            std::vector<int>& row_indA, std::vector<int>& col_indA,
                                            std::vector<double>& dB, std::vector<int>& row_indB,
                                            std::vector<int>& col_indB, std::vector<std::vector<double>>& out) {
  taskData->inputs_count.emplace_back(rowsA);
  taskData->inputs_count.emplace_back(columnsA);
  taskData->inputs_count.emplace_back(dA.size());
  taskData->inputs_count.emplace_back(row_indA.size());
  taskData->inputs_count.emplace_back(col_indA.size());

  taskData->inputs_count.emplace_back(rowsB);
  taskData->inputs_count.emplace_back(columnsB);
  taskData->inputs_count.emplace_back(dB.size());
  taskData->inputs_count.emplace_back(row_indB.size());
  taskData->inputs_count.emplace_back(col_indB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(row_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(col_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(dB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(row_indB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(col_indB.data()));

  for (size_t i = 0; i < out.size(); ++i) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out[i].data()));
  }
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs_count.emplace_back(out[0].size());
}