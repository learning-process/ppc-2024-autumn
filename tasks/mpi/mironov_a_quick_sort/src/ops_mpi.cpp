#include "mpi/mironov_a_quick_sort/include/ops_mpi.hpp"

#include <thread>
#include <utility>

bool mironov_a_quick_sort_mpi::QuickSortMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size();
    if (taskData->inputs_count[0] % world.size() > 0) {
      delta++;
    }

    input_ = std::vector<int>(delta * world.size(), std::numeric_limits<int>::max());
    result_.resize(input_.size(), std::numeric_limits<int>::max());
    int* it = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(it, it + taskData->inputs_count[0], input_.begin());
  }
  return true;
}

bool mironov_a_quick_sort_mpi::QuickSortMPI::validation() {
  internal_order_test();
  // Check count elements input & output
  if (world.rank() == 0)
    return (taskData->inputs_count[0] > 0) && (taskData->outputs_count[0] == taskData->inputs_count[0]);
  return true;
}

static void merge(std::vector<int>& vec, int start, int end, int* res) {
  if (res == nullptr) {
    return;
  }

  int ptr1 = 0;
  int ptr2 = start;
  int free = 0;

  while (ptr1 < start && ptr2 <= end) {
    if (vec[ptr1] <= vec[ptr2]) {
      res[free++] = vec[ptr1++];
    } else {
      res[free++] = vec[ptr2++];
    }
  }
  while (ptr1 < start) {
    res[free++] = vec[ptr1++];
  }
  while (ptr2 <= end) {
    res[free++] = vec[ptr2++];
  }

  for (int it = 0; it < free; ++it) {
    vec[it] = res[it];
  }
}

static void quickSort(std::vector<int>& arr, int start, int end) {
  if (start >= end) return;

  int pivot = arr[(start + end) / 2];
  int left = start;
  int right = end;

  while (left <= right) {
    while (arr[left] < pivot) left++;
    while (arr[right] > pivot) right--;

    if (left <= right) std::swap(arr[left++], arr[right--]);
  }

  quickSort(arr, start, right);
  quickSort(arr, left, end);
}

bool mironov_a_quick_sort_mpi::QuickSortMPI::run() {
  internal_order_test();

  broadcast(world, delta, 0);

  std::vector<int> local_input(delta);

  scatter(world, input_.data(), local_input.data(), delta, 0);
  quickSort(local_input, 0, delta - 1);

  boost::mpi::gather(world, local_input.data(), local_input.size(), result_.data(), 0);
  if (world.rank() == 0) {
    int* res = new int[input_.size()];
    for (int i = 1; i < world.size(); ++i) {
      merge(result_, i * delta, (i + 1) * delta - 1, res);
    }
    delete[] res;
  }

  return true;
}

bool mironov_a_quick_sort_mpi::QuickSortMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* it = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(result_.begin(), result_.begin() + taskData->outputs_count[0], it);
  }
  return true;
}

bool mironov_a_quick_sort_mpi::QuickSortSequential::pre_processing() {
  internal_order_test();

  // Init value for input and output
  input_ = std::vector<int>(taskData->inputs_count[0]);
  int* it = reinterpret_cast<int*>(taskData->inputs[0]);
  std::copy(it, it + taskData->inputs_count[0], input_.begin());
  return true;
}

bool mironov_a_quick_sort_mpi::QuickSortSequential::validation() {
  internal_order_test();
  // Check count elements input & output
  return (taskData->inputs_count[0] > 0) && (taskData->outputs_count[0] == taskData->inputs_count[0]);
}

bool mironov_a_quick_sort_mpi::QuickSortSequential::run() {
  internal_order_test();

  result_ = input_;
  quickSort(result_, 0, result_.size() - 1);
  return true;
}

bool mironov_a_quick_sort_mpi::QuickSortSequential::post_processing() {
  internal_order_test();
  int* it = reinterpret_cast<int*>(taskData->outputs[0]);
  std::copy(result_.begin(), result_.end(), it);
  return true;
}
