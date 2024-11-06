#include <limits>

#include "seq/agafeev_s_max_of_vector_elements/include/header.hpp"

// using namespace std::chrono_literals;
template <typename Type>
std::vector<Type> agafeev_s_max_of_vector_elements_sequental::genRandMatr(Type row_size, Type column_size) {
  auto rand_gen = std::mt19937(1337);
  std::vector<Type> matrix(row_size * column_size);
  for (int i = 0; i < matrix.size(); i++) matrix[i] = rand_gen() % 100;

  return matrix;
}

template <typename Type>
Type agafeev_s_max_of_vector_elements_sequental::getMaxValue(std::vector<Type> matrix) {
  Type max_result = std::numeric_limits<Type>::min();
  for (uint i = 0; i < matrix.size(); i++)
    if (max_result < matrix[i]) max_result = matrix[i];

  return max_result;
}

template <typename Type>
bool agafeev_s_max_of_vector_elements_sequental::MaxMatrixSequential<Type>::pre_processing() {
  internal_order_test();

  // Init value
  auto* temp_ptr = reinterpret_cast<Type*>(taskData->inputs[0]);
  input_.insert(input_.begin(), temp_ptr, temp_ptr + taskData->inputs_count[0]);

  return true;
}

template <typename Type>
bool agafeev_s_max_of_vector_elements_sequental::MaxMatrixSequential<Type>::validation() {
  internal_order_test();

  return taskData->outputs_count[0] == 1;
}

template <typename Type>
bool agafeev_s_max_of_vector_elements_sequental::MaxMatrixSequential<Type>::run() {
  internal_order_test();

  maxres_ = getMaxValue(input_);

  return true;
}

template <typename Type>
bool agafeev_s_max_of_vector_elements_sequental::MaxMatrixSequential<Type>::post_processing() {
  internal_order_test();

  maxres_ = reinterpret_cast<int*>(taskData->outputs[0])[0];

  return true;
}

template class agafeev_s_max_of_vector_elements_sequental::MaxMatrixSequential<int>;