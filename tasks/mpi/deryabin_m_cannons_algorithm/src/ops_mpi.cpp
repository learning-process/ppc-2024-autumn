#include "mpi/deryabin_m_cannons_algorithm/include/ops_mpi.hpp"

#include <thread>

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::pre_processing() {
  internal_order_test();
  input_matrix_A = std::vector<double>(taskData->inputs_count[0]);
  input_matrix_B = std::vector<double>(taskData->inputs_count[1]);
  auto* tmp_ptr_A = reinterpret_cast<double*>(taskData->inputs[0]);
  auto* tmp_ptr_B = reinterpret_cast<double*>(taskData->inputs[1]);
  std::copy(tmp_ptr_A, tmp_ptr_A + taskData->inputs_count[0], input_matrix_A.begin());
  std::copy(tmp_ptr_B, tmp_ptr_B + taskData->inputs_count[1], input_matrix_B.begin());
  output_matrix_C = std::vector<double>(input_matrix_A.size());
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] == taskData->inputs_count[1] ==
             pow((unsigned short)sqrt(taskData->inputs_count[0]), 2) &&
         taskData->outputs_count[0] == 1;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::run() {
  internal_order_test();
  unsigned short i = 0;
  unsigned short j = 0;
  unsigned short count = 0;
  auto dimension = (unsigned short)sqrt(input_matrix_A.size());
  while (i != dimension) {
    while (j != dimension) {
      while (count != dimension) {
        output_matrix_C[i * dimension + j] +=
            input_matrix_A[i * dimension + count] * input_matrix_B[count * dimension + j];
        count++;
      }
      j++;
    }
    i++;
  }
  return true;
}

bool deryabin_m_cannons_algorithm_mpi::CannonsAlgorithmMPITaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<double> *>(taskData->outputs[0])[0] = output_matrix_C;
  return true;
}
