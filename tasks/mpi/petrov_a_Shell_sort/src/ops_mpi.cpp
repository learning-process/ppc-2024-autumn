#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <stdexcept>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

bool TestTaskMPI::validation() {
  if (data_.empty()) {
    return false;
  }
  return true;
}

bool TestTaskMPI::pre_processing() {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    int vector_size = static_cast<int>(data_.size());
    int world_size = world.size();

    int base_size = vector_size / world_size;
    int remainder = vector_size % world_size;

    send_counts.resize(world_size);
    displacements.resize(world_size);

    for (int i = 0; i < world_size; ++i) {
      send_counts[i] = base_size + (i < remainder ? 1 : 0);
      displacements[i] = (i > 0) ? (displacements[i - 1] + send_counts[i - 1]) : 0;
    }

    for (int i = 1; i < world_size; ++i) {
      world.send(i, 0, data_.data() + displacements[i], send_counts[i]);
    }

    local_data.assign(data_.begin(), data_.begin() + send_counts[0]);
  } else {
    int local_size;
    world.recv(0, 0, local_size);
    local_data.resize(local_size);
    world.recv(0, 0, local_data.data(), local_size);
  }

  return true;
}

bool TestTaskMPI::run() {
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

  return true;
}

bool TestTaskMPI::post_processing() {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    data_ = local_data;

    for (int i = 1; i < world.size(); ++i) {
      std::vector<int> temp(send_counts[i]);
      world.recv(i, 1, temp.data(), temp.size());
      data_.insert(data_.end(), temp.begin(), temp.end());
    }

    std::sort(data_.begin(), data_.end());
  } else {
    world.send(0, 1, local_data.data(), local_data.size());
  }

  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
