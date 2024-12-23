#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <cstring>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::pre_processing() {
  size_t input_size = taskData->inputs_count[0];
  const auto* raw_data = reinterpret_cast<const unsigned char*>(taskData->inputs[0]);

  data_.resize(input_size);
  std::memcpy(data_.data(), raw_data, input_size * sizeof(int));

  return true;
}

bool TestTaskMPI::validation() {
  if (taskData->inputs.empty() || taskData->inputs_count.empty()) {
    return false;
  }

  if (taskData->outputs.empty() || taskData->outputs_count.empty()) {
    return false;
  }

  return true;
}

bool TestTaskMPI::run() {
  boost::mpi::communicator world;

  // Calculate local size and distribute data across processes
  int total_size = static_cast<int>(data_.size());
  int local_size = total_size / world.size();
  int remainder = total_size % world.size();

  std::vector<int> local_data(local_size + (world.rank() < remainder ? 1 : 0));
  boost::mpi::scatter(world, data_, local_data.data(), 0);

  // Local Shell sort
  for (int gap = local_data.size() / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < local_data.size(); ++i) {
      int temp = local_data[i];
      int j;
      for (j = i; j >= gap && local_data[j - gap] > temp; j -= gap) {
        local_data[j] = local_data[j - gap];
      }
      local_data[j] = temp;
    }
  }

  // Gather the sorted chunks at root
  boost::mpi::gather(world, local_data, data_, 0);

  // Root process merges the received sorted chunks
  if (world.rank() == 0) {
    std::inplace_merge(data_.begin(), data_.begin() + local_size, data_.end());
  }

  return true;
}

bool TestTaskMPI::post_processing() {
  if (taskData->outputs.empty() || taskData->outputs_count.empty()) {
    return false;
  }

  size_t output_size = taskData->outputs_count[0];
  auto* raw_output_data = reinterpret_cast<unsigned char*>(taskData->outputs[0]);
  std::memcpy(raw_output_data, data_.data(), output_size * sizeof(int));

  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
