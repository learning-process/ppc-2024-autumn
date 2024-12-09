// Copyright 2024 Nesterov Alexander
#include "seq/frolova_e_matrix_multiplication/include/ops_seq.hpp"

#include <thread>

using namespace std::chrono_literals;

void frolova_e_matrix_multiplication_seq::randomNumVec(int N, std::vector<int>& vec) {
  for (int i = 0; i < N; i++) {
    int num = rand() % 100 + 1;
    vec.push_back(num);
  }
}

std::vector<int> frolova_e_matrix_multiplication_seq::Multiplication(int M, int N, int K, const std::vector<int>& A, const std::vector<int>& B) {

  std::vector<int> C(M * N);

  for (int i = 0; i < M; ++i) {

    for (int j = 0; j < N; ++j) {

      C[i * N + j] = 0;

      for (int k = 0; k < K; ++k) 
          C[i * N + j] += A[i * K + k] * B[k * N + j];
    }
  }

  return C;
}

bool frolova_e_matrix_multiplication_seq::matrixMultiplication::pre_processing() {
  internal_order_test();

  // Init value for input and output
  int* value_1 = reinterpret_cast<int*>(taskData->inputs[0]);
  lineA = static_cast<size_t>(value_1[0]);
  columnA = static_cast<size_t>(value_1[1]);

  int* value_2 = reinterpret_cast<int*>(taskData->inputs[1]);
  lineB = static_cast<size_t>(value_2[0]);
  columnB = static_cast<size_t>(value_2[1]);

  int* matr1_ptr = reinterpret_cast<int*>(taskData->inputs[2]);
  matrixA.assign(matr1_ptr, matr1_ptr + taskData->inputs_count[2]);

  int* matr2_ptr = reinterpret_cast<int*>(taskData->inputs[3]);
  matrixB.assign(matr2_ptr, matr2_ptr + taskData->inputs_count[3]);

  matrixC.resize(lineA * columnB);

  return true;
}

bool frolova_e_matrix_multiplication_seq::matrixMultiplication::validation() {
  internal_order_test();
  // Check count elements of output

  int* value_1 = reinterpret_cast<int*>(taskData->inputs[0]);
  if (taskData->inputs_count[0] != 2) {
//    std::cout << "taskData->inputs_count[0] != 2" << std::endl;
    return false;
  }
  size_t line1 = static_cast<size_t>(value_1[0]);
  size_t column1 = static_cast<size_t>(value_1[1]);
  
  int* value_2 = reinterpret_cast<int*>(taskData->inputs[1]);
  if (taskData->inputs_count[1] != 2) {
//    std::cout << "taskData->inputs_count[1] != 2" << std::endl;
    return false;
  }
  size_t line2 = static_cast<size_t>(value_2[0]);
  size_t column2 = static_cast<size_t>(value_2[1]);

  if (value_1[1] != value_2[0]) {
//    std::cout << "value_1[1] != value_2[0]" << std::endl;
    return false;
  }
  if (taskData->inputs_count[2] != line1 * column1) {
//    std::cout << "taskData->inputs_count[2] != line1 * column1" << std::endl;
    return false;
  }
  if (taskData->inputs_count[3] != line2 * column2) {
//    std::cout << "taskData->inputs_count[3] != line2 * column2" << std::endl;
    return false;
  }
  if (taskData->outputs_count[0] != line1 * column2) {
//    std::cout << "taskData->outputs_count[0] != line1 * column2" << std::endl;
    return false;
  }

  return true;
           
}

bool frolova_e_matrix_multiplication_seq::matrixMultiplication::run() {
  internal_order_test();
  
  matrixC = Multiplication(lineA, columnB, columnA, matrixA, matrixB);

  return true;
}

bool frolova_e_matrix_multiplication_seq::matrixMultiplication::post_processing() {
  internal_order_test();
  
  for (int i = 0; i < lineA * columnB; i++) {
  
      reinterpret_cast<int*>(taskData->outputs[0])[i] = matrixC[i];

  }

  return true;
}
