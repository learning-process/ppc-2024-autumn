#include "mpi/petrov_a_Shell_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <iostream>
#include <numeric>
#include <vector>

namespace petrov_a_Shell_sort_mpi {

void shell_sort(std::vector<int>& local_data) {
  for (size_t gap = local_data.size() / 2; gap > 0; gap /= 2) {
    for (size_t i = gap; i < local_data.size(); ++i) {
      int temp = local_data[i];
      size_t j = i;
      while (j >= gap && local_data[j - gap] > temp) {
        local_data[j] = local_data[j - gap];
        j -= gap;
      }
      local_data[j] = temp;
    }
  }
}

void batcher_merge(std::vector<int>& data, int left_start, int right_start, int end) {
  std::inplace_merge(data.begin() + left_start, data.begin() + right_start, data.begin() + end);
  for (int step = 2; step <= (end - left_start); step *= 2) {
    for (int i = left_start; i + step <= end; i += step) {
      std::inplace_merge(data.begin() + i, data.begin() + i + step / 2, data.begin() + i + step);
    }
  }
}

void shell_sort_mpi(boost::mpi::communicator& world, std::vector<int>& data) {
  int world_rank = world.rank();
  int world_size = world.size();

  int vector_size = static_cast<int>(data.size());

  std::vector<int> local_data(vector_size / world_size + (world_rank < vector_size % world_size));
  std::vector<int> displs(world_size), send_counts(world_size);

  for (int i = 0; i < world_size; ++i) {
    send_counts[i] = vector_size / world_size + (i < vector_size % world_size);
    displs[i] = (i > 0) ? (displs[i - 1] + send_counts[i - 1]) : 0;
  }

  boost::mpi::scatterv(world, data.data(), send_counts, displs, local_data.data(), local_data.size(), 0);

  shell_sort(local_data);

  boost::mpi::gatherv(world, local_data.data(), local_data.size(), data.data(), send_counts, displs, 0);

  if (world_rank == 0) {
    for (int i = 1; i < world_size; ++i) {
      batcher_merge(data, 0, displs[i], displs[i] + send_counts[i]);
    }
  }
}

bool TestTaskMPI::validation() {
  boost::mpi::communicator world;
  if (world.rank() == 0 && data_.empty()) {
    std::cerr << "Input data is empty." << std::endl;
    return false;
  }
  return true;
}

bool TestTaskMPI::pre_processing() {
  boost::mpi::communicator world;
  int vector_size = static_cast<int>(data_.size());

  if (world.rank() == 0) {
    if (world.size() > vector_size) {
      std::cerr << "Number of processes exceeds the data size." << std::endl;
      return false;
    }
  }
  return true;
}

bool TestTaskMPI::run() {
  boost::mpi::communicator world;
  shell_sort_mpi(world, data_);
  return true;
}

bool TestTaskMPI::post_processing() {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    std::sort(data_.begin(), data_.end());
  }
  return true;
}

}  // namespace petrov_a_Shell_sort_mpi
