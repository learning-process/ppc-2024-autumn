#include "mpi/anufriev_d_star_topology/include/ops_mpi_anufriev.hpp"
#include <utility>

namespace anufriev_d_star_topology {

SimpleIntMPI::SimpleIntMPI(const std::shared_ptr<ppc::core::TaskData> taskData) : Task(std::move(taskData)) {}

bool SimpleIntMPI::pre_processing() {
  internal_order_test();
  size_t input_size = 0;
  if (world.rank() == 0) {
    input_size = taskData->inputs_count[0];
    for (int i = 1; i < world.size(); ++i) {
      world.send(i, 0, input_size);
    }
    total_size_ = input_size;
  } else {
    world.recv(0, 0, input_size);
    total_size_ = input_size;
  }

  if (world.rank() == 0) {
    input_data_.resize(input_size);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]),
              reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0], input_data_.begin());
  }
  distributeData();
  return true;
}

void SimpleIntMPI::distributeData() {
  size_t chunk_size = total_size_ / world.size();
  size_t remainder = total_size_ % world.size();

  size_t count = chunk_size + ((size_t)world.rank() < remainder ? 1 : 0);

  if (world.rank() == 0) {
    for (int i = 1; i < world.size(); ++i) {
      size_t start = i * chunk_size + std::min((size_t)i, remainder);
      size_t count_i = chunk_size + ((size_t)i < remainder ? 1 : 0);
      if (count_i > 0) {
        world.send(i, 0, input_data_.data() + start, count_i);
      }
    }
    size_t my_count = chunk_size + ((size_t)0 < remainder ? 1 : 0);
    if (my_count < total_size_) {
      input_data_.resize(my_count);
    }
  } else {
    if (count > 0) {
      input_data_.resize(count);
      world.recv(0, 0, input_data_.data(), count);
    } else {
      input_data_.resize(0);
    }
  }
}

bool SimpleIntMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count.empty() && taskData->outputs_count.empty() &&
           taskData->inputs_count[0] == taskData->outputs_count[0];
  }
  return true;
}

bool SimpleIntMPI::run() {
  internal_order_test();
  if (!input_data_.empty()) {
    for (int& value : input_data_) {
      value += world.rank();
    }
  }
  return true;
}

void SimpleIntMPI::gatherData() {
  size_t chunk_size = total_size_ / world.size();
  size_t remainder = total_size_ % world.size();

  size_t count = chunk_size + ((size_t)world.rank() < remainder ? 1 : 0);

  if (world.rank() == 0) {
    processed_data_.resize(total_size_);
    if (count > 0) {
      std::copy(input_data_.begin(), input_data_.end(), processed_data_.begin());
    }
    for (int i = 1; i < world.size(); ++i) {
      size_t receive_count = chunk_size + ((size_t)i < remainder ? 1 : 0);
      if (receive_count > 0) {
        std::vector<int> received_data(receive_count);
        world.recv(i, 0, received_data.data(), receive_count);
        size_t start_pos = i * chunk_size + std::min((size_t)i, remainder);
        std::copy(received_data.begin(), received_data.end(), processed_data_.begin() + start_pos);
      }
    }
  } else {
    if (count > 0) {
      world.send(0, 0, input_data_.data(), count);
    }
  }
}

bool SimpleIntMPI::post_processing() {
  internal_order_test();
  gatherData();
  if (world.rank() == 0) {
    std::copy(processed_data_.begin(), processed_data_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}

}  // namespace anufriev_d_star_topology