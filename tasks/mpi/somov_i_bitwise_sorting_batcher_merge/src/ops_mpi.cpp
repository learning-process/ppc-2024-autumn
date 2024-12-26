#include "mpi/somov_i_bitwise_sorting_batcher_merge/include/ops_mpi.hpp"

#include <algorithm>
#include <functional>
#include <random>
#include <vector>

void countingSortByByte(std::vector<uint64_t>& data, std::vector<uint64_t>& output, int byteIndex) {
  const int base = 256;
  std::array<int, base> count = {0};
  for (uint64_t num : data) {
    int byteValue = (num >> (byteIndex * 8)) & 0xFF;
    count[byteValue]++;
  }
  for (int i = 1; i < base; i++) {
    count[i] += count[i - 1];
  }
  for (int i = data.size() - 1; i >= 0; i--) {
    int byteValue = (data[i] >> (byteIndex * 8)) & 0xFF;
    output[--count[byteValue]] = data[i];
  }
}

void radixSort(std::vector<uint64_t>& data) {
  const int bytesCount = 8;
  std::vector<uint64_t> temp(data.size());

  for (int byteIndex = 0; byteIndex < bytesCount; byteIndex++) {
    countingSortByByte(data, temp, byteIndex);
    data.swap(temp);
  }
}

void radix_sort_double(std::vector<double>& arr) {
  size_t n = arr.size();
  if (n == 0) return;

  std::vector<double> positive;
  std::vector<double> negative;
  positive.reserve(n / 2);
  negative.reserve(n / 2);

  for (size_t i = 0; i < n; ++i) {
    if (arr[i] < 0) {
      negative.emplace_back(arr[i]);
    } else {
      positive.emplace_back(arr[i]);
    }
  }

  std::vector<uint64_t> posData(positive.size());
  for (size_t i = 0; i < positive.size(); ++i) {
    posData[i] = *reinterpret_cast<uint64_t*>(&positive[i]);
  }

  radixSort(posData);

  for (size_t i = 0; i < positive.size(); ++i) {
    positive[i] = *reinterpret_cast<double*>(&posData[i]);
  }

  std::vector<uint64_t> negData(negative.size());
  for (size_t i = 0; i < negative.size(); ++i) {
    negData[i] = *reinterpret_cast<uint64_t*>(&negative[i]);
    negData[i] = ~negData[i];
  }

  radixSort(negData);

  for (size_t i = 0; i < negative.size(); ++i) {
    negData[i] = ~negData[i];
    negative[i] = *reinterpret_cast<double*>(&negData[i]);
  }

  arr.clear();
  arr.insert(arr.end(), negative.begin(), negative.end());
  arr.insert(arr.end(), positive.begin(), positive.end());
}

bool somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_ = std::vector<double>(taskData->inputs_count[0]);
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
  return true;
}

bool somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] > 0;
}

bool somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  radix_sort_double(input_);
  res_ = input_;
  return true;
}

bool somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* tmp_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
  std::copy(res_.begin(), res_.end(), tmp_ptr);
  return true;
}

bool somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    input_.resize(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_.begin());
    res_.resize(taskData->inputs_count[0]);
  }
  return true;
}

bool somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0;
  }
  return true;
}

bool somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel::run() {
  internal_order_test();

  unsigned int delta = 0;
  unsigned int r = 0;
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    r = taskData->inputs_count[0] % world.size();
    local_input_.resize(delta + r);
  }

  broadcast(world, delta, 0);
  broadcast(world, r, 0);

  if (world.rank() != 0) local_input_.resize(delta);

  std::vector<int> local_size(world.size(), delta);
  local_size[0] += r;

  scatterv(world, input_, local_size, local_input_.data(), 0);

  radix_sort_double(local_input_);

  for (int c = 0; c < world.size() * 2; c++) {
    if (c % 2 == 0) {
      if (world.rank() % 2 == 0 && world.rank() + 1 != world.size()) {
        world.send(world.rank() + 1, c, local_input_);
        world.recv(world.rank() + 1, c, local_input_);
      }
      if (world.rank() % 2 != 0) {
        std::vector<double> tmp;
        world.recv(world.rank() - 1, c, tmp);
        std::vector<double> m(local_input_.size() + tmp.size());
        std::merge(local_input_.begin(), local_input_.end(), tmp.begin(), tmp.end(), m.begin());
        tmp.assign(m.begin(), m.begin() + tmp.size());
        local_input_.assign(m.end() - local_input_.size(), m.end());
        world.send(world.rank() - 1, c, tmp);
      }
    } else {
      if (world.rank() % 2 == 1 && world.rank() + 1 != world.size()) {
        world.send(world.rank() + 1, c, local_input_);
        world.recv(world.rank() + 1, c, local_input_);
      }
      if (world.rank() % 2 == 0 && world.rank() != 0) {
        std::vector<double> tmp;
        world.recv(world.rank() - 1, c, tmp);
        std::vector<double> m(local_input_.size() + tmp.size());
        std::merge(local_input_.begin(), local_input_.end(), tmp.begin(), tmp.end(), m.begin());
        tmp.assign(m.begin(), m.begin() + tmp.size());
        local_input_.assign(m.end() - local_input_.size(), m.end());
        world.send(world.rank() - 1, c, tmp);
      }
    }
  }

  gatherv(world, local_input_.data(), local_input_.size(), res_.data(), local_size, 0);
  return true;
}

bool somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    auto* tmp_ptr = reinterpret_cast<double*>(taskData->outputs[0]);
    std::copy(res_.begin(), res_.end(), tmp_ptr);
  }
  return true;
}
