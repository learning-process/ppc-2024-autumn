#include "mpi/tarakanov_d_ring_topology/include/ops_mpi.hpp"

namespace tarakanov_d_test_task_mpi {

bool TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world_comm_.rank() == 0) {
    int* input_data = reinterpret_cast<int*>(taskData->inputs.front());
    size_t input_size = taskData->inputs_count.front();
    data_buffer_.clear();
    data_buffer_.reserve(input_size);
    for (size_t i = 0; i < input_size; ++i) {
      data_buffer_.push_back(input_data[i]);
    }
  }

  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();

  if (world_comm_.rank() != 0) {
    return true;
  }

  bool is_valid = (taskData->inputs.size() == 1) && (taskData->outputs.size() == 1) &&
                  (taskData->inputs_count.front() == taskData->outputs_count.front());

  return is_valid;
}

bool TestMPITaskParallel::run() {
  internal_order_test();

  int current_rank = world_comm_.rank();
  int total_processes = world_comm_.size();

  if (total_processes <= 1) {
    return true;
  }

  if (current_rank == 0) {
    int dest_rank = current_rank + 1;
    buffer_size_ = static_cast<int>(data_buffer_.size());
    world_comm_.send(dest_rank, 0, buffer_size_);
    world_comm_.send(dest_rank, 1, data_buffer_);
  }

  int source_rank = (current_rank == 0) ? (total_processes - 1) : (current_rank - 1);
  world_comm_.recv(source_rank, 0, received_size_);
  data_buffer_.resize(received_size_);
  world_comm_.recv(source_rank, 1, data_buffer_);

  if (current_rank != 0) {
    int dest_rank = (current_rank == (total_processes - 1)) ? 0 : (current_rank + 1);
    buffer_size_ = static_cast<int>(data_buffer_.size());
    world_comm_.send(dest_rank, 0, buffer_size_);
    world_comm_.send(dest_rank, 1, data_buffer_);
  }

  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world_comm_.rank() == 0) {
    int* output_data = reinterpret_cast<int*>(taskData->outputs.front());
    std::copy(data_buffer_.begin(), data_buffer_.end(), output_data);
  }

  return true;
}

}  // namespace tarakanov_d_test_task_mpi