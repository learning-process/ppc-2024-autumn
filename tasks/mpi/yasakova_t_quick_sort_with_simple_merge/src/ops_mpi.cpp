#include "mpi/yasakova_t_quick_sort_with_simple_merge/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <queue>
#include <stack>

namespace yasakova_t_quick_sort_with_simple_merge_mpi {

struct PriorityQueueNode {
  int key;
  int origin_rank;

  bool operator>(const PriorityQueueNode& other) const { return key > other.key; }
};

void mpi_worker_function(boost::mpi::communicator& world, const std::vector<int>& local_data) {
  int size = local_data.size();

  if (!local_data.empty()) {
    world.send(0, 0, local_data[0]);
  }

  for (int i = 1; i < size; ++i) {
    world.recv(0, 1);
    world.send(0, 0, local_data[i]);
  }
}

std::vector<int> master_function(boost::mpi::communicator& world, const std::vector<int>& local_data,
                                 const std::vector<int>& element_sizes) {
  std::priority_queue<PriorityQueueNode, std::vector<PriorityQueueNode>, std::greater<>> min_heap;
  std::vector<int> remaining_element_sizes = element_sizes;
  std::vector<int> result;
  int iter = 0;

  min_heap.push({local_data[iter++], 0});
  remaining_element_sizes[0]--;
  for (int i = 1; i < world.size(); ++i) {
    if (remaining_element_sizes[i] > 0) {
      int key;
      world.recv(i, 0, key);
      min_heap.push({key, i});
      remaining_element_sizes[i]--;
    }
  }

  while (!min_heap.empty()) {
    PriorityQueueNode node = min_heap.top();
    min_heap.pop();
    result.push_back(node.key);
    if (node.origin_rank == 0 && remaining_element_sizes[0] > 0) {
      min_heap.push({local_data[iter++], node.origin_rank});
      remaining_element_sizes[0]--;
    } else {
      if (remaining_element_sizes[node.origin_rank] > 0) {
        world.send(node.origin_rank, 1);
        int next_key;
        world.recv(node.origin_rank, 0, next_key);
        min_heap.push({next_key, node.origin_rank});
        remaining_element_sizes[node.origin_rank]--;
      }
    }
  }

  return result;
}

void mpi_merge_function(boost::mpi::communicator& world, const std::vector<int>& local_data,
                        const std::vector<int>& element_sizes, std::vector<int>& res) {
  if (world.rank() == 0) {
    res = master_function(world, local_data, element_sizes);
  } else {
    mpi_worker_function(world, local_data);
  }
  world.barrier();
}

void quicksort_iterative(std::vector<int>& data) {
  std::stack<std::pair<int, int>> stack;
  stack.emplace(0, data.size() - 1);

  while (!stack.empty()) {
    auto [low, high] = stack.top();
    stack.pop();

    if (low < high) {
      int pivot = data[high];
      int i = low - 1;

      for (int j = low; j < high; ++j) {
        if (data[j] < pivot) {
          std::swap(data[++i], data[j]);
        }
      }
      std::swap(data[i + 1], data[high]);
      int p = i + 1;

      stack.emplace(low, p - 1);
      stack.emplace(p + 1, high);
    }
  }
}

bool SimpleMergeQuicksort::validation() {
  internal_order_test();

  return *reinterpret_cast<int*>(taskData->inputs[0]) > 0;
}

bool SimpleMergeQuicksort::pre_processing() {
  internal_order_test();

  size_of_vector = *reinterpret_cast<int*>(taskData->inputs[0]);

  if (world.rank() == 0) {
    auto* vec_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int vec_size = taskData->inputs_count[1];

    original_vector.assign(vec_data, vec_data + vec_size);
  }
  element_sizes.resize(world.size());
  displacement.resize(world.size());

  for (int i = 0; i < world.size(); ++i) {
    element_sizes[i] = size_of_vector / world.size() + (i < size_of_vector % world.size() ? 1 : 0);
    displacement[i] = (i == 0) ? 0 : displacement[i - 1] + element_sizes[i - 1];
  }

  partitioned_vector.resize(element_sizes[world.rank()]);

  return true;
}

bool SimpleMergeQuicksort::run() {
  internal_order_test();

  boost::mpi::scatterv(world, original_vector.data(), element_sizes, displacement, partitioned_vector.data(),
                       element_sizes[world.rank()], 0);

  quicksort_iterative(partitioned_vector);

  mpi_merge_function(world, partitioned_vector, element_sizes, original_vector);

  return true;
}

bool SimpleMergeQuicksort::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* out_vector = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(original_vector.begin(), original_vector.end(), out_vector);
  }

  return true;
}

}  // namespace yasakova_t_quick_sort_with_simple_merge_mpi