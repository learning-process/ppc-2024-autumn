#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::pre_processing() {
  size_t input_size = taskData->inputs_count[0];
  const auto* raw_data = reinterpret_cast<const unsigned char*>(taskData->inputs[0]);

  data_.resize(input_size);
  memcpy(data_.data(), raw_data, input_size * sizeof(int));

  return true;
}

bool TestTaskMPI::validation() {
  if (taskData->inputs.empty()) {
    std::cerr << "Validation failed: inputs are empty!" << std::endl;
    return false;
  }

  if (taskData->inputs_count.empty()) {
    std::cerr << "Validation failed: inputs_count is empty!" << std::endl;
    return false;
  }

  if (taskData->outputs.empty()) {
    std::cerr << "Validation failed: outputs are empty!" << std::endl;
    return false;
  }

  if (taskData->outputs_count.empty()) {
    std::cerr << "Validation failed: outputs_count is empty!" << std::endl;
    return false;
  }

  if (taskData->inputs_count.size() != taskData->outputs_count.size()) {
    std::cerr << "Validation failed: inputs_count size does not match outputs_count size!" << std::endl;
    return false;
  }

  return true;
}

bool TestTaskMPI::run() {
  int n = data_.size();

  for (int gap = n / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < n; ++i) {
      int temp = data_[i];
      int j;
      for (j = i; j >= gap && data_[j - gap] > temp; j -= gap) {
        data_[j] = data_[j - gap];
      }
      data_[j] = temp;
    }
  }

  return true;
}

bool TestTaskMPI::post_processing() {
  size_t output_size = taskData->outputs_count[0];
  auto* raw_output_data = reinterpret_cast<unsigned char*>(taskData->outputs[0]);
  memcpy(raw_output_data, data_.data(), output_size * sizeof(int));
  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
