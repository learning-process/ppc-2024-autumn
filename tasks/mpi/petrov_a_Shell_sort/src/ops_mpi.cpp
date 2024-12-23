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

  // Calculate the total size and divide work across processes
  int total_size = static_cast<int>(data_.size());
  int local_size = total_size / world.size() + (world.rank() < (total_size % world.size()) ? 1 : 0);

  std::vector<int> local_data(local_size);

  if (world.rank() == 0) {
    // Flatten the data for scatterv
    std::vector<int> send_counts(world.size(), total_size / world.size());
    for (int i = 0; i < total_size % world.size(); ++i) {
      send_counts[i]++;
    }

    std::vector<int> displacements(world.size(), 0);
    for (size_t i = 1; i < displacements.size(); ++i) {
      displacements[i] = displacements[i - 1] + send_counts[i - 1];
    }

    boost::mpi::scatterv(world, data_, send_counts, displacements, local_data, 0);
  } else {
    boost::mpi::scatterv(world, local_data, 0);
  }

  // Local Shell sort
  for (int gap = local_data.size() / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < local_data.size(); ++i) {
      int temp = local_data[i];
      size_t j;
      for (j = i; j >= gap && local_data[j - gap] > temp; j -= gap) {
        local_data[j] = local_data[j - gap];
      }
      local_data[j] = temp;
    }
  }

  // Gather sorted chunks
  if (world.rank() == 0) {
    std::vector<int> recv_counts(world.size(), total_size / world.size());
    for (int i = 0; i < total_size % world.size(); ++i) {
      recv_counts[i]++;
    }

    std::vector<int> displacements(world.size(), 0);
    for (size_t i = 1; i < displacements.size(); ++i) {
      displacements[i] = displacements[i - 1] + recv_counts[i - 1];
    }

    boost::mpi::gatherv(world, local_data, data_, recv_counts, displacements, 0);
  } else {
    boost::mpi::gatherv(world, local_data, 0);
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
