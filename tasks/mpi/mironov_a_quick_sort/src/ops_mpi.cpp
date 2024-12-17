#include "mpi/mironov_a_quick_sort/include/ops_mpi.hpp"

#include <stack>
#include <thread>
#include <utility>

//#define DEBUG

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
    input_ = std::vector < int>(delta * world.size(), std::numeric_limits<int>::max());
    result_ = std::vector<int>(taskData->inputs_count[0]);
    int* it = reinterpret_cast<int*>(taskData->inputs[0]);
#ifdef DEBUG
    cout << "sz " << input_.size() << " " << result_.size() << endl;
#endif
    for(size_t i = 0; i < taskData->inputs_count[0]; ++i) {
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

static void merge(std::vector<int>& vec1, std::vector<int>& vec2) {
  std::vector<int> res;
  res.reserve(vec1.size() + vec2.size());
  
  int ptr1 = 0;
  int ptr2 = 0;

  while (ptr1 < vec1.size() && ptr2 < vec2.size()) {
    if (vec1[ptr1] <= vec2[ptr2]) {
      res.push_back(vec1[ptr1++]);
    } else {
      res.push_back(vec2[ptr2++]);
    }
  }
  while (ptr1 < vec1.size()) {
    res.push_back(vec1[ptr1++]);
  }
  while (ptr2 < vec2.size()) {
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

  if (world.rank() == 0) {
    // probably better to use isend
    for (int proc = 1; proc < world.size(); proc++) {
      world.send(proc, 0, input_.data() + proc * delta, delta);
    }
  };

  std::vector<int> local_input;
  if (world.rank() == 0) {
    local_input = std::vector<int>(input_.begin(), input_.begin() + delta);
  } else {
    local_input.resize(delta);
    world.recv(0, 0, local_input.data(), delta);
  }
  // cout << "Stage 1 " <<  world.rank() << endl;
  quickSort(local_input, 0, delta - 1);
  int world_sz = world.size();

  if (world.rank() == 0) {
    if (world_sz == 1) {
      result_ = std::move(local_input);
      return true;
    }
    int curr_rank = 1;
    std::vector<int> vec2;
    while (curr_rank < world_sz) {

      vec2.resize(local_input.size());
#ifdef DEBUG
      cout << "recv: " << curr_rank << " " << vec2.size() << " " << local_input.size() << " " << world.rank() << endl;
#endif
      world.recv(curr_rank, 0, vec2.data(), local_input.size());
      merge(local_input, vec2);
#ifdef DEBUG  
      cout << "MERGED " << local_input.size() << endl;
#endif
      curr_rank <<= 1;
    }
    result_ = std::move(local_input);

  } else {
    int rank = world.rank();
    int curr_rank = world.rank();
    int need_rank = 1;

    std::vector<int> vec2;

    while (!(curr_rank % 2)) {
      vec2.resize(local_input.size());
      if (rank + need_rank < world_sz) {
        world.recv(rank + need_rank, 0, vec2.data(), local_input.size());
        merge(local_input, vec2);
#ifdef DEBUG 
        cout << "Abort: " << rank + need_rank << " " << local_input.size() << " " << world.rank() << endl;
        cout << "MERGED_$ " << local_input.size() << " " << world.rank() << endl;
#endif
      }
      else
      {
        local_input.resize(local_input.size() * 2, std::numeric_limits<int>::max());
      }
      need_rank <<= 1;
      curr_rank >>= 1;
    }
#ifdef DEBUG 
    if (world.rank() == 4)
    {
      cout << "WOWWW: " << need_rank << " " << curr_rank << endl;
    }
    int c = world.rank() - need_rank;
 
    cout << "C: " << c << " " << local_input.size() << " " << world.rank() << endl;
#endif
    world.send(world.rank() - need_rank, 0, local_input.data(), local_input.size());
  }

  return true;
}

bool mironov_a_quick_sort_mpi::QuickSortMPI::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    int* it = reinterpret_cast<int*>(taskData->outputs[0]);
#ifdef DEBUG  
    cout << "OUTPUT: " <<  taskData->outputs_count[0] << " " << result_.size() << endl;
#endif
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
