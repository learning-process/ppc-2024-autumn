#include "mpi/mironov_a_quick_sort/include/ops_mpi.hpp"

#include <stack>
#include <thread>
#include <utility>

// #define DEBUG

#ifdef DEBUG
using namespace std;
#endif
bool mironov_a_quick_sort_mpi::QuickSortMPI::pre_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    delta = taskData->inputs_count[0] / world.size() + (taskData->inputs_count[0] % world.size() > 0);
#ifdef DEBUG
    cout << "Delta " << delta << endl;
#endif
    input_ = std::vector<int>(delta * world.size(), std::numeric_limits<int>::max());
    result_ = std::vector<int>(taskData->inputs_count[0]);
    int* it = reinterpret_cast<int*>(taskData->inputs[0]);
#ifdef DEBUG
    cout << "sz " << input_.size() << " " << result_.size() << endl;
#endif
    for (size_t i = 0; i < taskData->inputs_count[0]; ++i) {
      input_[i] = it[i];
    }
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

static void merge(std::vector<int>& vec1, std::vector<int>& vec2, int start, int end) {
  std::vector<int> res;
  res.reserve(vec1.size() + end - start + 1);

  int ptr1 = 0;
  int ptr2 = start;

  while (ptr1 < vec1.size() && ptr2 <= end) {
    if (vec1[ptr1] <= vec2[ptr2]) {
      res.push_back(vec1[ptr1++]);
    } else {
      res.push_back(vec2[ptr2++]);
    }
  }
  while (ptr1 < vec1.size()) {
    res.push_back(vec1[ptr1++]);
  }
  while (ptr2 <= end) {
    res.push_back(vec2[ptr2++]);
  }
  vec1 = std::move(res);
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
  std::vector<int> merged;

  scatter(world, input_.data(), local_input.data(), delta, 0);
  quickSort(local_input, 0, delta - 1);

  if (world.rank() == 0) {
    merged.resize(input_.size());
  }

  boost::mpi::gather(world, local_input.data(), local_input.size(), merged.data(), 0);
  if (world.rank() == 0) {
#ifdef DEBUG

    for (auto x : merged) {
      std::cout << x << " ";
    }
    std::cout << " 1231 " << std::endl;
    
    for (auto x : result_) {
      std::cout << x << " ";
    }
    std::cout << " 1231wwW " << std::endl;
#endif  // DEBUG
    result_ = std::move(local_input);
    for (int i = 1; i < world.size(); ++i) {
#ifdef DEBUG
      std::cout << "! " << i * delta << " " << (i + 1) * delta - 1 << std::endl;
#endif  // DEBUG

      
      merge(result_, merged, i * delta, (i + 1) * delta - 1);
    }
#ifdef DEBUG
    for (auto x : result_) {
      std::cout << x << " ";
    }
    std::cout << " 1231wwweqweW " << std::endl;
#endif  // DEBUG
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
