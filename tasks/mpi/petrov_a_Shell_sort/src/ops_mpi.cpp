#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::pre_processing() {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    size_t input_size = taskData->inputs_count[0];
    const auto* raw_data = reinterpret_cast<const unsigned char*>(taskData->inputs[0]);

    data_.resize(input_size);
    std::memcpy(data_.data(), raw_data, input_size * sizeof(int));
  }
  return true;
}

bool TestTaskMPI::validation() {
  boost::mpi::communicator world;
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }
  if (world.rank() == 0 && (!taskData->outputs.empty() && !taskData->outputs_count.empty())) {
    return false;
  }
  return true;
}

bool TestTaskMPI::run() {
  boost::mpi::communicator world;
  size_t local_size = data_.size() / world.size();
  size_t remainder = data_.size() % world.size();

  std::vector<int> local_data;
  if (world.rank() == 0) {
    for (int i = 1; i < world.size(); ++i) {
      size_t start_idx = i * local_size + std::min(static_cast<size_t>(i), remainder);
      size_t chunk_size = local_size + (i < remainder ? 1 : 0);
      world.send(i, 0, std::vector<int>(data_.begin() + start_idx, data_.begin() + start_idx + chunk_size));
    }
    local_data.assign(data_.begin(), data_.begin() + local_size + (remainder > 0 ? 1 : 0));
  } else {
    world.recv(0, 0, local_data);
  }

  for (size_t gap = local_data.size() / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < local_data.size(); ++i) {
      int temp = local_data[i];
      size_t j;
      for (j = i; j >= gap && local_data[j - gap] > temp; j -= gap) {
        local_data[j] = local_data[j - gap];
      }
      local_data[j] = temp;
    }
  }

  if (world.rank() != 0) {
    std::vector<int> received_data;
    world.recv(0, 0, received_data);
  } else {
    world.send(1, 0, local_data);
  }


  return true;
}

bool TestTaskMPI::post_processing() {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    size_t output_size = taskData->outputs_count[0];
    auto* raw_output_data = reinterpret_cast<unsigned char*>(taskData->outputs[0]);
    std::memcpy(raw_output_data, data_.data(), output_size * sizeof(int));
  }
  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
