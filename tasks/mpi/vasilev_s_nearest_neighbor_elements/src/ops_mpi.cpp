#include "mpi/vasilev_s_nearest_neighbor_elements/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI::pre_processing() {
  internal_order_test();
  input_.resize(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());

  min_diff_ = std::numeric_limits<int>::max();
  index1_ = -1;
  index2_ = -1;
  return true;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI::validation() {
  internal_order_test();
  return !taskData->inputs_count.empty() && taskData->inputs_count[0] >= 2 && taskData->outputs_count[0] == 3;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI::run() {
  internal_order_test();
  for (size_t i = 0; i < input_.size() - 1; ++i) {
    int diff = std::abs(input_[i + 1] - input_[i]);
    if (diff < min_diff_) {
      min_diff_ = diff;
      index1_ = static_cast<int>(i);
      index2_ = static_cast<int>(i + 1);
    }
  }
  return true;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsSequentialMPI::post_processing() {
  internal_order_test();
  int* output_ptr = reinterpret_cast<int*>(taskData->outputs[0]);
  output_ptr[0] = min_diff_;
  output_ptr[1] = index1_;
  output_ptr[2] = index2_;
  return true;
}

std::pair<std::vector<int>, std::vector<int>> vasilev_s_nearest_neighbor_elements_mpi::partitionArray(
    int amount, int num_partitions) {
  std::vector<int> displs(num_partitions);
  std::vector<int> sizes(num_partitions);
  int total_elements = amount + num_partitions - 1;
  int base_size = total_elements / num_partitions;
  int extra_elements = total_elements % num_partitions;

  if (amount <= num_partitions) {
    for (int i = 0; i < num_partitions; i++) {
      if (i < amount - 1) {
        sizes[i] = 2;
        displs[i] = i;
      } else {
        sizes[i] = 0;
        displs[i] = -1;
      }
    }
  } else {
    for (int i = 0; i < num_partitions; i++) {
      if (extra_elements > 0) {
        sizes[i] = base_size + 1;
        extra_elements--;
      } else {
        sizes[i] = base_size;
      }

      if (i == 0) {
        displs[i] = 0;
      } else {
        displs[i] = displs[i - 1] + sizes[i - 1] - 1;
      }
    }
  }
  return {displs, sizes};
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI::validation() {
  internal_order_test();
  return world.rank() != 0 || taskData->inputs_count[0] > 1;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI::pre_processing() {
  internal_order_test();

  rank_offset_ = 0;

  if (world.rank() == 0) {
    min_diff_ = std::numeric_limits<int>::max();
    index1_ = -1;
    index2_ = -1;
    std::tie(displacement, distribution) =
        vasilev_s_nearest_neighbor_elements_mpi::partitionArray(taskData->inputs_count[0], world.size());
  }

  return true;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI::run() {
  internal_order_test();

  unsigned int amount = 0;

  if (world.rank() == 0) {
    amount = taskData->inputs_count[0];
  }

  boost::mpi::broadcast(world, amount, 0);

  boost::mpi::broadcast(world, displacement, 0);
  boost::mpi::broadcast(world, distribution, 0);

  rank_offset_ = displacement[world.rank()];

  input_.resize(distribution[world.rank()]);
  if (world.rank() == 0) {
    const auto* in_p = reinterpret_cast<int*>(taskData->inputs[0]);
    boost::mpi::scatterv(world, in_p, distribution, displacement, input_.data(), distribution[0], 0);
  } else {
    boost::mpi::scatterv(world, input_.data(), distribution[world.rank()], 0);
  }

  LocalResult local_result{std::numeric_limits<int>::max(), -1, -1};
  const std::size_t size = input_.size();

  if (size > 0) {
    for (size_t i = 0; i < input_.size() - 1; ++i) {
      int diff = std::abs(input_[i + 1] - input_[i]);

      int current_index1 = static_cast<int>(rank_offset_ + i);
      int current_index2 = static_cast<int>(rank_offset_ + i + 1);

      if (diff < local_result.min_diff || (diff == local_result.min_diff && current_index1 < local_result.index1)) {
        local_result.min_diff = diff;
        local_result.index1 = current_index1;
        local_result.index2 = current_index2;
      }
    }
  }

  LocalResult global_result{std::numeric_limits<int>::max(), -1, -1};
  boost::mpi::reduce(world, local_result, global_result, boost::mpi::minimum<LocalResult>(), 0);

  if (world.rank() == 0) {
    min_diff_ = global_result.min_diff;
    index1_ = global_result.index1;
    index2_ = global_result.index2;
  }
  return true;
}

bool vasilev_s_nearest_neighbor_elements_mpi::FindClosestNeighborsParallelMPI::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = min_diff_;
    reinterpret_cast<int*>(taskData->outputs[0])[1] = index1_;
    reinterpret_cast<int*>(taskData->outputs[0])[2] = index2_;
  }

  return true;
}
