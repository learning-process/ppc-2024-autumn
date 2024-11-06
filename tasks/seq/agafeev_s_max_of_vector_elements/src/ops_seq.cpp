#include "seq/agafeev_s_max_of_vector_elements/include/ops_seq.hpp"

std::vector<int> agafeev_s_max_of_vector_elements_sequental::create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(1337);
  std::vector<int> matrix(row_size * column_size);
  for (uint i = 0; i < matrix.size(); i++) matrix[i] = rand_gen() % 100;

  return matrix;
}

int agafeev_s_max_of_vector_elements_sequental::get_MaxValue(std::vector<int> matrix) {
  int max_result = std::numeric_limits<int>::min();
  for (uint i = 0; i < matrix.size(); i++)
    if (max_result < matrix[i]) max_result = matrix[i];

  return max_result;
}

bool agafeev_s_max_of_vector_elements_sequental::MaxMatrixSequential::pre_processing() {
  internal_order_test();

  // Init value
  auto* temp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  input_.insert(input_.begin(), temp_ptr, temp_ptr + taskData->inputs_count[0]);

  return true;
}

bool agafeev_s_max_of_vector_elements_sequental::MaxMatrixSequential::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

bool agafeev_s_max_of_vector_elements_sequental::MaxMatrixSequential::run() {
  internal_order_test();

  maxres_ = get_MaxValue(input_);

  return true;
}

bool agafeev_s_max_of_vector_elements_sequental::MaxMatrixSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<int*>(taskData->outputs[0])[0] = maxres_;

  return true;
}