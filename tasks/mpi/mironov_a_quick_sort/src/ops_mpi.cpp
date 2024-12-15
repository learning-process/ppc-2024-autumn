#include "mpi/mironov_a_quick_sort/include/ops_mpi.hpp"

#include <stack>
#include <thread>
#include <utility>

using namespace std::chrono_literals;

bool mironov_a_quick_sort_mpi::QuickSortMPI::pre_processing() {
  internal_order_test();

  return true;
}

bool mironov_a_quick_sort_mpi::QuickSortMPI::validation() {
  internal_order_test();
  // Check count elements input & output
  return (taskData->inputs_count[0] > 0) && (taskData->outputs_count[0] == taskData->inputs_count[0]);
}

bool mironov_a_quick_sort_mpi::QuickSortMPI::run() {
  internal_order_test();
  return true;
}

bool mironov_a_quick_sort_mpi::QuickSortMPI::post_processing() {
  internal_order_test();
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
