#include "mpi/kazunin_n_quicksort_simple_merge/include/ops_mpi.hpp"

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <queue>
#include <stack>

namespace kazunin_n_quicksort_simple_merge_mpi {

struct HeapNode {
  int value;
  int source_rank;

  bool operator>(const HeapNode& other) const { return value > other.value; }
};

void worker_function(boost::mpi::communicator& world, const std::vector<int>& local_data) {
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
                                 const std::vector<int>& sizes) {
  std::priority_queue<HeapNode, std::vector<HeapNode>, std::greater<>> min_heap;
  std::vector<int> remaining_sizes = sizes;
  std::vector<int> result;
  int iter = 0;

  min_heap.push({local_data[iter++], 0});
  remaining_sizes[0]--;
  for (int i = 1; i < world.size(); ++i) {
    if (remaining_sizes[i] > 0) {
      int value;
      world.recv(i, 0, value);
      min_heap.push({value, i});
      remaining_sizes[i]--;
    }
  }

  while (!min_heap.empty()) {
    HeapNode node = min_heap.top();
    min_heap.pop();
    result.push_back(node.value);
    if (node.source_rank == 0 && remaining_sizes[0] > 0) {
      min_heap.push({local_data[iter++], node.source_rank});
      remaining_sizes[0]--;
    } else {
      if (remaining_sizes[node.source_rank] > 0) {
        world.send(node.source_rank, 1);
        int next_value;
        world.recv(node.source_rank, 0, next_value);
        min_heap.push({next_value, node.source_rank});
        remaining_sizes[node.source_rank]--;
      }
    }
  }

  return result;
}

void merge_func(boost::mpi::communicator& world, const std::vector<int>& local_data, const std::vector<int>& sizes,
                std::vector<int>& res) {
  if (world.rank() == 0) {
    res = master_function(world, local_data, sizes);
  } else {
    worker_function(world, local_data);
  }
  world.barrier();
}

void iterative_quicksort(std::vector<int>& data) {
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

bool QuicksortSimpleMerge::validation() {
  internal_order_test();

  return *reinterpret_cast<int*>(taskData->inputs[0]) > 0;
}

bool QuicksortSimpleMerge::pre_processing() {
  internal_order_test();

  vector_size = *reinterpret_cast<int*>(taskData->inputs[0]);

  if (world.rank() == 0) {
    auto* vec_data = reinterpret_cast<int*>(taskData->inputs[1]);
    int vec_size = taskData->inputs_count[1];

    input_vector.assign(vec_data, vec_data + vec_size);
  }
  sizes.resize(world.size());
  displs.resize(world.size());

  for (int i = 0; i < world.size(); ++i) {
    sizes[i] = vector_size / world.size() + (i < vector_size % world.size() ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + sizes[i - 1];
  }

  local_vector.resize(sizes[world.rank()]);

  return true;
}

bool QuicksortSimpleMerge::run() {
  internal_order_test();

  boost::mpi::scatterv(world, input_vector.data(), sizes, displs, local_vector.data(), sizes[world.rank()], 0);

  iterative_quicksort(local_vector);

  merge_func(world, local_vector, sizes, input_vector);

  return true;
}

bool QuicksortSimpleMerge::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    auto* out_vector = reinterpret_cast<int*>(taskData->outputs[0]);
    std::copy(input_vector.begin(), input_vector.end(), out_vector);
  }

  return true;
}

}  // namespace kazunin_n_quicksort_simple_merge_mpi
