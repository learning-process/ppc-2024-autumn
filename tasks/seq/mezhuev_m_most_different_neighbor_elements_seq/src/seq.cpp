#include "seq/mezhuev_m_most_different_neighbor_elements_seq/include/seq.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace mezhuev_m_most_different_neighbor_elements {

bool MostDifferentNeighborElements::validation() {
  internal_order_test();
  if (!taskData) {
    std::cerr << "TaskData is not set!" << std::endl;
    return false;
  }

  if (taskData->inputs.empty()) {
    std::cerr << "Inputs are empty!" << std::endl;
    return false;
  }

  if (taskData->inputs_count.empty() || taskData->inputs_count[0] < 2) {
    std::cerr << "Inputs count is less than 2!" << std::endl;
    return false;
  }

  if (taskData->outputs_count.empty() || taskData->outputs_count[0] != 2) {
    return false;
  }

  int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  return std::adjacent_find(input_ptr, input_ptr + taskData->inputs_count[0], std::not_equal_to<>()) !=
         input_ptr + taskData->inputs_count[0];
}

bool MostDifferentNeighborElements::pre_processing() {
  internal_order_test();
  if (!taskData || taskData->inputs.empty() || taskData->inputs_count.empty() || taskData->inputs_count[0] < 2) {
    return false;
  }

  int* input_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  input.assign(input_ptr, input_ptr + taskData->inputs_count[0]);
  result.resize(2);
  return true;
}

bool MostDifferentNeighborElements::run() {
  internal_order_test();
  if (input.size() < 2) {
    return false;
  }

  int max_difference = 0;
  bool found = false;

  for (size_t i = 0; i < input.size() - 1; ++i) {
    int current_difference = std::abs(input[i] - input[i + 1]);
    if (!found || current_difference > max_difference) {
      max_difference = current_difference;
      result[0] = input[i];
      result[1] = input[i + 1];
      found = true;
    }
  }

  if (!found) {
    result[0] = input[0];
    result[1] = input[0];
  }

  return true;
}

bool MostDifferentNeighborElements::post_processing() {
  internal_order_test();
  if (!taskData || taskData->outputs.empty() || result.size() != 2) {
    return false;
  }

  int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result.begin(), result.end(), output_ptr);
  return true;
}

}  // namespace mezhuev_m_most_different_neighbor_elements