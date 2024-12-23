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

  if (!taskData->outputs.empty() && !taskData->outputs_count.empty()) {
    return false;
  }

  return true;
}

bool TestTaskMPI::run() {
  boost::mpi::communicator world;
  int local_size = data_.size() / world.size();
  std::vector<int> local_data(local_size);

  world.scatter(data_, local_data, 0);

  for (int gap = local_size / 2; gap > 0; gap /= 2) {
    for (int i = gap; i < local_size; ++i) {
      int temp = local_data[i];
      int j;
      for (j = i; j >= gap && local_data[j - gap] > temp; j -= gap) {
        local_data[j] = local_data[j - gap];
      }
      local_data[j] = temp;
    }
  }

  world.gather(0, local_data, data_);
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
