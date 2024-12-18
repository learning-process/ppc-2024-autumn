// Copyright 2023 Nesterov Alexander
#include "mpi/kondratev_ya_radix_sort_batcher_merge/include/ops_mpi.hpp"

void kondratev_ya_radix_sort_batcher_merge_mpi::radixSortDouble(std::vector<double>& arr, int32_t start, int32_t end) {
  constexpr int32_t byte_size = 8;
  constexpr int32_t double_bit_size = byte_size * sizeof(double);
  constexpr int32_t full_byte_bit_mask = 0xFF;
  constexpr uint64_t sign_bit_mask = 1ULL << (double_bit_size - 1);
  int32_t byte_range = std::pow(2, byte_size);

  int32_t size = end - start + 1;
  std::vector<double> temp(size);

  // uint64_t because sizeof(uint64_t) == sizeof(double)
  auto* bits = reinterpret_cast<uint64_t*>(arr.data() + start);

  // Invert the sign bit for negative numbers
  for (int32_t i = 0; i < size; i++) {
    if ((bool)(bits[i] >> (double_bit_size - 1))) {
      bits[i] = ~bits[i];
    } else {
      bits[i] |= sign_bit_mask;
    }
  }

  // Sorting by bits
  for (int32_t shift = 0; shift < double_bit_size; shift += byte_size) {
    std::vector<int32_t> count(byte_range, 0);
    for (int32_t i = 0; i < size; i++) {
      count[(bits[i] >> shift) & full_byte_bit_mask]++;
    }

    std::partial_sum(count.begin(), count.end(), count.begin());

    for (int32_t i = size - 1; i >= 0; i--) {
      int32_t bucket = (bits[i] >> shift) & full_byte_bit_mask;
      temp[count[bucket] - 1] = arr[start + i];
      count[bucket]--;
    }
    std::copy(temp.begin(), temp.end(), arr.begin() + start);
  }

  // Restore inverted signs
  for (int32_t i = 0; i < size; i++) {
    if (!(bool)(bits[i] & sign_bit_mask)) {
      bits[i] = ~bits[i];
    } else {
      bits[i] &= ~sign_bit_mask;
    }
  }
}

bool kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();

  auto* input = reinterpret_cast<double*>(taskData->inputs[0]);
  data_.assign(input, input + taskData->inputs_count[0]);

  return true;
}

bool kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskSequential::validation() {
  internal_order_test();

  return !taskData->inputs_count.empty() && taskData->inputs_count[0] > 0 && !taskData->outputs_count.empty() &&
         !taskData->outputs_count.empty() && taskData->outputs_count[0] == taskData->inputs_count[0] &&
         !taskData->inputs.empty() && taskData->inputs[0] != nullptr && !taskData->outputs.empty() &&
         taskData->outputs[0] != nullptr;
}

bool kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskSequential::run() {
  internal_order_test();

  radixSortDouble(data_, 0, data_.size() - 1);

  return true;
}

bool kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();

  auto* output = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(data_.begin(), data_.end(), output);

  return true;
}

bool kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* pointer = reinterpret_cast<double*>(taskData->inputs[0]);
    size_ = taskData->inputs_count[0];
    input_.assign(pointer, pointer + size_);
    res_.resize(size_);
  }

  return true;
}

bool kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel::validation() {
  internal_order_test();

  return (world.rank() != 0) ||
         (!taskData->inputs_count.empty() && taskData->inputs_count[0] > 0 && !taskData->outputs_count.empty() &&
          !taskData->outputs_count.empty() && taskData->outputs_count[0] == taskData->inputs_count[0] &&
          !taskData->inputs.empty() && taskData->inputs[0] != nullptr && !taskData->outputs.empty() &&
          taskData->outputs[0] != nullptr);
}
bool kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  broadcast(world, size_, 0);

  int32_t step = size_ / world.size();
  int32_t remain = size_ % world.size();

  std::vector<int32_t> sizes(world.size(), step);
  for (int32_t i = 0; i < remain; i++) sizes[i]++;

  local_input_.resize(sizes[world.rank()]);
  scatterv(world, input_, sizes, local_input_.data(), 0);

  kondratev_ya_radix_sort_batcher_merge_mpi::radixSortDouble(local_input_, 0, local_input_.size() - 1);

  for (int32_t merge_step = 0; merge_step < world.size(); ++merge_step) {
    if (merge_step % 2 == 0) {
      if (world.rank() % 2 == 0 && world.rank() + 1 < world.size()) {
        exchange_and_merge(world.rank(), sizes[world.rank()], world.rank() + 1, sizes[world.rank() + 1]);
      } else if (world.rank() % 2 != 0) {
        exchange_and_merge(world.rank() - 1, sizes[world.rank() - 1], world.rank(), sizes[world.rank()]);
      }
    } else {
      if (world.rank() % 2 == 1 && world.rank() + 1 < world.size()) {
        exchange_and_merge(world.rank(), sizes[world.rank()], world.rank() + 1, sizes[world.rank() + 1]);
      } else if (world.rank() % 2 == 0 && world.rank() > 0) {
        exchange_and_merge(world.rank() - 1, sizes[world.rank() - 1], world.rank(), sizes[world.rank()]);
      }
    }
  }

  gatherv(world, local_input_.data(), local_input_.size(), res_.data(), sizes, 0);

  return true;
}

void kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel::exchange_and_merge(int32_t rank1, int32_t size1,
                                                                                        int32_t rank2, int32_t size2) {
  if (size1 <= 0 || size2 <= 0) return;
  std::vector<double> neighbor_data;

  boost::mpi::request reqs[2];
  if (world.rank() == rank1) {
    neighbor_data.resize(size2);
    reqs[0] = world.isend(rank2, 0, local_input_.data(), size1);
    reqs[1] = world.irecv(rank2, 0, neighbor_data.data(), size2);
  } else if (world.rank() == rank2) {
    neighbor_data.resize(size1);
    reqs[0] = world.irecv(rank1, 0, neighbor_data.data(), size1);
    reqs[1] = world.isend(rank1, 0, local_input_.data(), size2);
  }

  reqs[0].wait();
  reqs[1].wait();

  if (!neighbor_data.empty()) {
    std::vector<double> merged_data(local_input_.size() + neighbor_data.size());
    merge(local_input_, neighbor_data, merged_data);

    if (world.rank() == rank1) {
      local_input_.assign(merged_data.begin(), merged_data.begin() + local_input_.size());
    } else {
      local_input_.assign(merged_data.end() - local_input_.size(), merged_data.end());
    }
  }
}

void kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel::merge(const std::vector<double>& first,
                                                                           const std::vector<double>& second,
                                                                           std::vector<double>& result) {
  auto it1 = first.begin();
  auto last1 = first.end();
  auto it2 = second.begin();
  auto last2 = second.end();
  auto out = result.begin();

  while (it1 != last1 && it2 != last2) {
    if (*it1 <= *it2)
      *out++ = *it1++;
    else
      *out++ = *it2++;
  }

  while (it1 != last1) {
    *out++ = *it1++;
  }

  while (it2 != last2) {
    *out++ = *it2++;
  }
}

bool kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* pointer = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), pointer);
  }

  return true;
}
