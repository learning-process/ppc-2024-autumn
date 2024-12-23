#include "mpi/gordeeva_t_shell_sort_batcher_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <vector>

using namespace std::chrono_literals;

void gordeeva_t_shell_sort_batcher_merge_mpi::shellSort(std::vector<int>& arr) {
  size_t arr_length = arr.size();
  for (size_t step = arr_length / 2; step > 0; step /= 2) {
    for (size_t i = step; i < arr_length; i++) {
      size_t j = i;
      while (j >= step && arr[j - step] > arr[j]) {
        std::swap(arr[j], arr[j - step]);
        j -= step;
      }
    }
  }
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  size_t sz = taskData->inputs_count[0];
  auto* input_tmp = reinterpret_cast<int32_t*>(taskData->inputs[0]);
  input_.assign(input_tmp, input_tmp + sz);
  res_.resize(sz);

  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
  if (taskData->inputs_count[0] <= 0) return false;
  if (taskData->outputs_count.size() != 1) return false;
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  shellSort(input_);
  res_ = input_;
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  int* output_matr = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), output_matr);
  return true;
}

void gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::batcher_merge(size_t rank1, size_t rank2,
                                                                                 std::vector<int>& local_input_local) {
  size_t rank = world.rank();
  std::vector<int> received_data;

  if (rank == rank1) {
    world.send(rank2, 0, local_input_local);
    world.recv(rank2, 0, received_data);
  } else if (rank == rank2) {
    world.recv(rank1, 0, received_data);
    world.send(rank1, 0, local_input_local);
  }

  if (!received_data.empty()) {
    std::vector<int> merged_data(local_input_local.size() + received_data.size());
    std::merge(local_input_local.begin(), local_input_local.end(), received_data.begin(), received_data.end(),
               merged_data.begin());

    if (rank == rank1) {
      local_input_local.assign(merged_data.begin(), merged_data.begin() + local_input_local.size());
    } else if (rank == rank2) {
      local_input_local.assign(merged_data.end() - local_input_local.size(), merged_data.end());
    }
  }
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    sz_mpi = taskData->inputs_count[0];
    auto* input_tmp = reinterpret_cast<int32_t*>(taskData->inputs[0]);
    input_.assign(input_tmp, input_tmp + sz_mpi);
    res_.resize(sz_mpi);
  }
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty()) return false;
    if (taskData->inputs_count[0] <= 0) return false;
    if (taskData->outputs_count.size() != 1) return false;
  }
  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  size_t rank = world.rank();
  size_t size = world.size();

  size_t sz_mpi_local = 0;
  boost::mpi::broadcast(world, sz_mpi, 0);
  sz_mpi_local = sz_mpi;

  size_t delta = sz_mpi_local / size;
  size_t ost = sz_mpi_local % size;

  std::vector<int> sz_rk(size, static_cast<int>(delta));

  for (size_t i = 0; i < ost; ++i) {
    sz_rk[i]++;
  }

  if (rank != 0) {
    input_.resize(sz_mpi_local);
  }
  boost::mpi::broadcast(world, input_.data(), sz_mpi_local, 0);

  std::vector<int> local_input(sz_rk[rank]);
  boost::mpi::scatterv(world, input_, sz_rk, local_input.data(), 0);

  shellSort(local_input);

  for (size_t i = 0; i < size; ++i) {
    if (rank % 2 == i % 2 && rank + 1 < size) {
      batcher_merge(rank, rank + 1, local_input);
    } else if (rank % 2 != i % 2 && rank > 0) {
      batcher_merge(rank - 1, rank, local_input);
    }
  }

  if (rank == 0) {
    res_.resize(sz_mpi_local);
  }
  boost::mpi::gatherv(world, local_input.data(), local_input.size(), res_.data(), sz_rk, 0);

  return true;
}

bool gordeeva_t_shell_sort_batcher_merge_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0 && !taskData->outputs.empty()) {
    std::copy(res_.begin(), res_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}
