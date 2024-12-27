#include "mpi/belov_a_radix_sort_with_batcher_mergesort/include/ops_mpi.hpp"

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "boost/mpi/detail/broadcast_sc.hpp"

using namespace std;

namespace belov_a_radix_batcher_mergesort_mpi {

int RadixBatcherMergesortParallel::get_number_digit_capacity(bigint num) {
  return (num == 0 ? 1 : static_cast<int>(log10(abs(num))) + 1);
}

void RadixBatcherMergesortParallel::sort(vector<bigint>& arr) {
  vector<bigint> pos, neg;

  for (const auto& num : arr) {
    (num >= 0 ? pos : neg).push_back(abs(num));
  }

  radix_sort(pos, false);
  radix_sort(neg, true);

  arr.clear();
  arr.reserve(neg.size() + pos.size());
  for (const auto& num : neg) arr.push_back(-num);
  arr.insert(arr.end(), pos.begin(), pos.end());
}

void RadixBatcherMergesortParallel::radix_sort(vector<bigint>& arr, bool invert) {
  if (arr.empty()) return;

  bigint max_val = *std::max_element(arr.begin(), arr.end());
  int max_val_digit_capacity = get_number_digit_capacity(max_val);
  int iter = 1;

  for (bigint digit_place = 1; iter <= max_val_digit_capacity; digit_place *= 10, ++iter) {
    counting_sort(arr, digit_place);
  }

  if (invert) std::reverse(arr.begin(), arr.end());
}

void RadixBatcherMergesortParallel::counting_sort(vector<bigint>& arr, bigint digit_place) {
  vector<bigint> output(arr.size());
  int count[10] = {};

  for (const auto& num : arr) {
    bigint index = (num / digit_place) % 10;
    count[index]++;
  }

  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  for (int i = arr.size() - 1; i >= 0; i--) {
    bigint num = arr[i];
    bigint index = (num / digit_place) % 10;
    output[--count[index]] = num;
  }

  std::copy(output.begin(), output.end(), arr.begin());
}

bool RadixBatcherMergesortParallel::pre_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    n = taskData->inputs_count[0];
    auto* input_array_data = reinterpret_cast<bigint*>(taskData->inputs[0]);
    array.assign(input_array_data, input_array_data + n);
  }
  return true;
}

bool RadixBatcherMergesortParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (taskData->inputs.size() == 1 && !(taskData->inputs_count.size() < 2) && taskData->inputs_count[0] != 0 &&
            (taskData->inputs_count[0] == taskData->inputs_count[1]) && !taskData->outputs.empty());
  }
  return true;
}

vector<int> calculate_sizes(size_t input_size, size_t num_proc) {
  size_t count = input_size / num_proc;
  size_t mod = input_size % num_proc;
  vector<int> sizes(num_proc, count);

  transform(sizes.cbegin(), sizes.cbegin() + mod, sizes.begin(), [](auto i) { return i + 1; });

  return sizes;
}

vector<int> calculate_displ(size_t input_size, size_t num_proc) {
  auto sizes = calculate_sizes(input_size, num_proc);
  vector<int> displ(num_proc, 0);

  for (size_t i = 1; i < num_proc; ++i) {
    displ[i] = displ[i - 1] + sizes[i - 1];
  }
  return displ;
}

vector<bigint> RadixBatcherMergesortParallel::merge(const vector<bigint>& arr1, const vector<bigint>& arr2) {
  vector<bigint> merged;
  merged.reserve(arr1.size() + arr2.size());

  size_t i = 0, j = 0;

  while (i < arr1.size() && j < arr2.size()) {
    if (arr1[i] <= arr2[j]) {
      merged.push_back(arr1[i++]);
    } else {
      merged.push_back(arr2[j++]);
    }
  }

  while (i < arr1.size()) {
    merged.push_back(arr1[i++]);
  }

  while (j < arr2.size()) {
    merged.push_back(arr2[j++]);
  }

  return merged;
}

void RadixBatcherMergesortParallel::compare_and_swap(vector<int>& data, int i, int j) {
  if (data[i] > data[j]) {
    std::swap(data[i], data[j]);
  }
}

void RadixBatcherMergesortParallel::odd_even_merge(vector<int>& data, int low, int high) {
  int size = high - low + 1;
  if (size <= 1) return;
  int mid = size / 2;

  odd_even_merge(data, low, low + mid - 1);
  odd_even_merge(data, low + mid, high);

  for (int i = low; i + mid <= high; ++i) {
    compare_and_swap(data, i, i + mid);
  }
}

bool RadixBatcherMergesortParallel::run() {
  internal_order_test();
  boost::mpi::broadcast(world, n, 0);

  int power_of_2 = 1;
  while (power_of_2 * 2 <= world.size()) {
    power_of_2 *= 2;
  }

  const bool participating = world.rank() < power_of_2;

  if (!participating) {
    world.split(1);
    return true;
  }

  auto com = world.split(0);
  const int rank = com.rank();
  const int com_size = com.size();

  vector<bigint> local_data;

  vector<int> sizes(power_of_2);
  vector<int> displ(power_of_2);

  if (com.rank() == 0) {
    sizes = calculate_sizes(n, power_of_2);
    displ = calculate_displ(n, power_of_2);

    local_data.resize(sizes[0]);
    local_data.assign(array.begin(), array.begin() + sizes[0]);
    boost::mpi::scatterv(com, array, sizes, displ, local_data.data(), local_data.size(), 0);
  } else {
    int cur_size = rank < n % com_size ? (n / com_size + 1) : (n / com_size);
    local_data.resize(cur_size);
    boost::mpi::scatterv(com, local_data.data(), cur_size, 0);
  }

  sort(local_data);

  for (int step = 0; step < com.size(); ++step) {
    if ((step + com.rank()) % 2 == 0) {
      if (com.rank() < com.size() - 1) {
        vector<bigint> neighbor_data;
        world.send(world.rank() + 1, 0, local_data);
        world.recv(world.rank() + 1, 0, neighbor_data);
        local_data = merge(local_data, neighbor_data);
      }
    } else {
      if (com.rank() > 0) {
        vector<bigint> neighbor_data;
        world.recv(world.rank() - 1, 0, neighbor_data);
        world.send(world.rank() - 1, 0, local_data);
        local_data = merge(neighbor_data, local_data);
      }
    }
  }

  if (com.rank() == 0) {
    array.clear();
    array.insert(array.begin(), local_data.begin(), local_data.end());
    for (int i = 1; i < com.size(); i++) {
      com.recv(i, 1, local_data);
      array.insert(array.end(), local_data.begin(), local_data.end());
    }
  } else {
    com.send(0, 1, local_data);
  }

  return true;
}

bool RadixBatcherMergesortParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    copy(array.begin(), array.begin() + n, reinterpret_cast<bigint*>(taskData->outputs[0]));
  }

  return true;
}

int RadixBatcherMergesortSequential::get_number_digit_capacity(bigint num) {
  return (num == 0 ? 1 : static_cast<int>(log10(abs(num))) + 1);
}

void RadixBatcherMergesortSequential::sort(vector<bigint>& arr) {
  vector<bigint> pos, neg;

  for (const auto& num : arr) {
    (num >= 0 ? pos : neg).push_back(abs(num));
  }

  radix_sort(pos, false);
  radix_sort(neg, true);

  arr.clear();
  arr.reserve(neg.size() + pos.size());
  for (const auto& num : neg) arr.push_back(-num);
  arr.insert(arr.end(), pos.begin(), pos.end());
}

void RadixBatcherMergesortSequential::radix_sort(vector<bigint>& arr, bool invert) {
  if (arr.empty()) return;

  bigint max_val = *std::max_element(arr.begin(), arr.end());
  int max_val_digit_capacity = get_number_digit_capacity(max_val);
  int iter = 1;

  for (bigint digit_place = 1; iter <= max_val_digit_capacity; digit_place *= 10, ++iter) {
    counting_sort(arr, digit_place);
  }

  if (invert) std::reverse(arr.begin(), arr.end());
}

void RadixBatcherMergesortSequential::counting_sort(vector<bigint>& arr, bigint digit_place) {
  vector<bigint> output(arr.size());
  int count[10] = {};

  for (const auto& num : arr) {
    bigint index = (num / digit_place) % 10;
    count[index]++;
  }

  for (int i = 1; i < 10; i++) {
    count[i] += count[i - 1];
  }

  for (int i = arr.size() - 1; i >= 0; i--) {
    bigint num = arr[i];
    bigint index = (num / digit_place) % 10;
    output[--count[index]] = num;
  }

  std::copy(output.begin(), output.end(), arr.begin());
}

bool RadixBatcherMergesortSequential::pre_processing() {
  internal_order_test();

  n = taskData->inputs_count[0];
  auto* input_array_data = reinterpret_cast<bigint*>(taskData->inputs[0]);
  array.assign(input_array_data, input_array_data + n);

  return true;
}

bool RadixBatcherMergesortSequential::validation() {
  internal_order_test();

  return (taskData->inputs.size() == 1 && !(taskData->inputs_count.size() < 2) && taskData->inputs_count[0] != 0 &&
          (taskData->inputs_count[0] == taskData->inputs_count[1]) && !taskData->outputs.empty());
}

bool RadixBatcherMergesortSequential::run() {
  internal_order_test();

  sort(array);
  return true;
}

bool RadixBatcherMergesortSequential::post_processing() {
  internal_order_test();

  copy(array.begin(), array.end(), reinterpret_cast<bigint*>(taskData->outputs[0]));
  return true;
}

}  // namespace belov_a_radix_batcher_mergesort_mpi