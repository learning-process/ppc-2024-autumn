#include "mpi/anufriev_d_star_topology/include/ops_mpi_anufriev.hpp"

namespace anufriev_d_star_topology {

SimpleIntMPI::SimpleIntMPI(std::shared_ptr<ppc::core::TaskData> taskData) : Task(taskData) {}

bool SimpleIntMPI::pre_processing() {
  internal_order_test();
  size_t input_size = 0;
  if (world.rank() == 0) {
    input_size = taskData->inputs_count[0];  // Получаем размер на rank 0
  }
  boost::mpi::broadcast(world, input_size, 0);
  input_data_.resize(input_size);
  if (world.rank() == 0) {
    // Предполагаем, что входные данные - это один вектор int
    //input_data_.resize(taskData->inputs_count[0]);
    std::copy(reinterpret_cast<int*>(taskData->inputs[0]),
              reinterpret_cast<int*>(taskData->inputs[0]) + taskData->inputs_count[0], input_data_.begin());
  }
  distributeData();
  return true;
}

void SimpleIntMPI::distributeData() {
  if (world.rank() == 0) {
    for (int i = 1; i < world.size(); ++i) {
      world.send(i, 0, input_data_);
    }
  } else {
    world.recv(0, 0, input_data_);
  }
}

bool SimpleIntMPI::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count.size() == 1 && taskData->outputs_count.size() == 1 &&
           taskData->inputs_count[0] == taskData->outputs_count[0];
  }
  return true;
}

bool SimpleIntMPI::run() {
  internal_order_test();
  for (int& value : input_data_) {
    value += world.rank();
  }
  return true;
}

void SimpleIntMPI::gatherData() {
  if (world.rank() == 0) {
    processed_data_.resize(input_data_.size());
    for (size_t i = 0; i < input_data_.size(); ++i) {
      processed_data_[i] = input_data_[i];
    }

    std::vector<int> received_data(input_data_.size());
    for (int i = 1; i < world.size(); ++i) {
      world.recv(i, 0, received_data);
      for (size_t j = 0; j < input_data_.size(); ++j) {
        processed_data_[j] += received_data[j];
      }
    }
  } else {
    world.send(0, 0, input_data_);
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