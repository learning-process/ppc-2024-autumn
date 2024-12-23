#include "seq/deryabin_m_cannons_algorithm/include/ops_seq.hpp"

#include <thread>

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::pre_processing() {
  internal_order_test();
  input_matrix_A = reinterpret_cast<std::vector<double> *>(taskData->inputs[0])[0];
  input_matrix_B = reinterpret_cast<std::vector<double> *>(taskData->inputs[1])[0];
  output_matrix_C = std::vector<double>(input_matrix_A.size());
  return true;
}

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::validation() {
  internal_order_test();
  return (taskData->inputs[0])[0].size() == (taskData->inputs[1])[0].size() ==
             pow((unsigned short)sqrt(input_matrix_A.size()), 2) &&
         taskData->outputs_count[0] == 1;
}

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::run() {
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

bool deryabin_m_cannons_algorithm_seq::CannonsAlgorithmTaskSequential::post_processing() {
  internal_order_test();
  reinterpret_cast<std::vector<double> *>(taskData->outputs[0])[0] = output_matrix_C;
  return true;
}  
