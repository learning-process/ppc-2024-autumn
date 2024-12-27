
#include "seq/morozov_e_mult_sparse_matrix/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;
template <typename T>
void morozov_e_mult_sparse_matrix::printMatrix(std::vector<std::vector<T>> m) {
  for (size_t i = 0; i < m.size(); ++i) {
    for (size_t j = 0; j < m[0].size(); ++j) {
      std::cout << m[i][j] << " ";
    }
    std::cout << "\n";
  }
}
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

  // Инициализация col_pointers
  col_pointers.resize(num_cols + 1, 0);

  // Подсчет количества ненулевых элементов в каждом столбце
  for (int col = 0; col < num_cols; ++col) {
    for (int row = 0; row < num_rows; ++row) {
      if (matrix[row][col] != 0) {
        col_pointers[col + 1]++;
      }
    }
  }
  // printVector(col_pointers);
  //  Преобразование счетчиков в указатели
  for (int col = 0; col < num_cols; ++col) {
    col_pointers[col + 1] += col_pointers[col];
  }
  // printVector(col_pointers);
  // Заполнение values и row_indices
  for (int col = 0; col < num_cols; ++col) {
    for (int row = 0; row < num_rows; ++row) {
      double value = matrix[row][col];
      if (value != 0) {
        // int index = col_pointers[col];
        values.push_back(value);
        row_indices.push_back(row);
        col_pointers[col]++;
      }
    }
  }

  // Восстановление col_pointers
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

  // morozov_e_mult_sparse_matrix::printMatrix(ans);
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
  convertToBasicMatrixs(dA, row_indA, col_indA, dB, row_indB, col_indB, rowsA, columnsA, rowsB, columnsB);
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
