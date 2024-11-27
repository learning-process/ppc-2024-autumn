#include "mpi/anufriev_d_star_topology/include/ops_mpi_anufriev.hpp"

namespace anufriev_d_star_topology {

SimpleIntMPI::SimpleIntMPI(std::shared_ptr<ppc::core::TaskData> taskData) : Task(taskData) {}

bool SimpleIntMPI::pre_processing() {
  internal_order_test();
  size_t input_size = 0;
  if (world.rank() == 0) {
    input_size = taskData->inputs_count[0];
  }
  boost::mpi::broadcast(world, input_size, 0);
  input_data_.resize(input_size);
  if (world.rank() == 0) {
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]),
              reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0], input_data_.begin());
  }
  distributeData();
  return true;
}

void SimpleIntMPI::distributeData() {
  size_t total_size = input_data_.size();
  size_t chunk_size = total_size / world.size();
  size_t remainder = total_size % world.size();

  size_t count;

  if (world.rank() == 0) {
    for (int i = 1; i < world.size(); ++i) {
      size_t start = i * chunk_size + std::min((size_t)i, remainder);
      count = chunk_size + (static_cast<size_t>(i) < remainder ? 1 : 0);
      world.send(i, 0, input_data_.data() + start, count);
    }
    input_data_.resize(chunk_size + (static_cast<size_t>(0) < remainder ? 1 : 0));
  } else {
    if (input_data_.empty()) {
      return;
    }
    count = chunk_size + (static_cast<size_t>(world.rank()) < remainder ? 1 : 0);
    input_data_.resize(count);
    world.recv(0, 0, input_data_.data(), input_data_.size());
  }
}

bool SimpleIntMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count.size() > 0 && taskData->outputs_count.size() > 0 &&
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
  size_t total_size = taskData->outputs_count[0];
  size_t chunk_size = total_size / world.size();
  size_t remainder = total_size % world.size();

  std::vector<int> received_data(chunk_size + (remainder > 0));

  if (world.rank() == 0) {
    processed_data_.resize(total_size);
    if (!input_data_.empty()) {
      std::copy(input_data_.begin(), input_data_.end(), processed_data_.begin());
    }

    for (int i = 1; i < world.size(); ++i) {
      size_t receive_count = chunk_size + (static_cast<size_t>(i) < remainder);

      world.recv(i, 0, received_data.data(), receive_count);
      size_t start_pos = i * chunk_size + std::min((size_t)i, remainder);
      std::copy(received_data.begin(), received_data.begin() + receive_count, processed_data_.begin() + start_pos);
    }

  } else {
    if (!input_data_.empty()) {
      world.send(0, 0, input_data_.data(), input_data_.size());
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