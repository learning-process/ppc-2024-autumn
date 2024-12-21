#include "mpi/mezhuev_m_most_different_neighbor_elements/include/mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cmath>
#include <stdexcept>

namespace mezhuev_m_most_different_neighbor_elements {

bool MostDifferentNeighborElements::validation() {
  internal_order_test();
  if (!taskData || taskData->inputs.empty() || taskData->outputs.empty()) {
    return false;
  }

  if (taskData->inputs.size() != 1 || taskData->outputs.size() != 2) {
    return false;
  }

  if (taskData->inputs[0] == nullptr || taskData->outputs[0] == nullptr || taskData->outputs[1] == nullptr) {
    return false;
  }

  if (taskData->inputs_count.empty() || taskData->outputs_count.empty()) {
    return false;
  }

  if (taskData->inputs_count[0] == 0 || taskData->outputs_count[0] == 0) {
    return false;
  }

  return true;
}

bool MostDifferentNeighborElements::pre_processing() {
  internal_order_test();

  if (taskData == nullptr) {
    std::cerr << "Error: taskData is nullptr" << std::endl;
    return false;
  }

  if (taskData->inputs_count.empty()) {
    std::cerr << "Error: inputs_count is empty" << std::endl;
    return false;
  }

  if (taskData->inputs_count[0] == 0) {
    std::cerr << "Error: inputs_count[0] is 0" << std::endl;
    return false;
  }

  if (taskData->inputs.size() != 1) {
    std::cerr << "Error: inputs.size() is not 1" << std::endl;
    return false;
  }

  size_t data_size = taskData->inputs_count[0];
  result_[0] = 0;
  result_[1] = 0;

  if (data_size == 0) {
    std::cerr << "Error: data_size is 0" << std::endl;
    return false;
  }

  return true;
}

bool MostDifferentNeighborElements::run() {
  internal_order_test();

  int* input_data = reinterpret_cast<int*>(taskData->inputs[0]);
  int* output_data_1 = reinterpret_cast<int*>(taskData->outputs[0]);
  int* output_data_2 = reinterpret_cast<int*>(taskData->outputs[1]);
  size_t data_size = taskData->inputs_count[0];

  int rank = world.rank();
  int size = world.size();

  if (input_data == nullptr || output_data_1 == nullptr || output_data_2 == nullptr || data_size == 0 ||
      static_cast<size_t>(size) > data_size) {
    return true;
  }

  size_t rows_per_process = data_size / size;
  size_t extra_rows = data_size % size;
  size_t start_index = rank * rows_per_process + std::min(rank, static_cast<int>(extra_rows));
  size_t end_index = (rank + 1) * rows_per_process + std::min(rank + 1, static_cast<int>(extra_rows));

  int local_max_diff = 0;
  int local_max_index = 0;
  for (size_t i = start_index; i < end_index - 1; ++i) {
    int diff = std::abs(input_data[i + 1] - input_data[i]);
    if (diff > local_max_diff) {
      local_max_diff = diff;
      local_max_index = i;
    }
  }

  int global_max_diff = 0;
  int global_max_index = 0;

  if (rank == 0) {
    global_max_diff = local_max_diff;
    global_max_index = local_max_index;

    for (int i = 1; i < size; ++i) {
      int recv_diff = 0;
      int recv_index = 0;

      world.recv(i, 0, recv_diff);
      world.recv(i, 1, recv_index);

      if (recv_diff > global_max_diff) {
        global_max_diff = recv_diff;
        global_max_index = recv_index;
      }
    }

    output_data_1[0] = std::min(input_data[global_max_index], input_data[global_max_index + 1]);
    output_data_2[0] = std::max(input_data[global_max_index], input_data[global_max_index + 1]);
  } else {
    world.send(0, 0, local_max_diff);
    world.send(0, 1, local_max_index);
  }

  return true;
}

bool MostDifferentNeighborElements::post_processing() {
  internal_order_test();
  if (!taskData || taskData->outputs[0] == nullptr || taskData->outputs[1] == nullptr) {
    return false;
  }

  for (size_t i = 0; i < taskData->outputs_count[0]; ++i) {
    if (taskData->outputs[0][i] != 0) {
      return true;
    }
  }
  return false;
}

}  // namespace mezhuev_m_most_different_neighbor_elements