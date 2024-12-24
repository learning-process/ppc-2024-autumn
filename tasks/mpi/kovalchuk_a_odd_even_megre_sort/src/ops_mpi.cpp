#include "mpi/kovalchuk_a_odd_even_megre_sort/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/serialization/vector.hpp>
#include <functional>
#include <random>
#include <string>
#include <vector>

void batcher_merge(std::vector<int>& array, int start, int mid, int end) {
  int n = end - start;
  if (n <= 1) return;

  std::vector<int> even_array;
  std::vector<int> odd_array;
  for (int i = start; i < end; ++i) {
    if (i % 2 == start % 2) {
      even_array.push_back(array[i]);
    } else {
      odd_array.push_back(array[i]);
    }
  }

  batcher_merge(even_array, 0, even_array.size() / 2, even_array.size());
  batcher_merge(odd_array, 0, odd_array.size() / 2, odd_array.size());

  std::merge(even_array.begin(), even_array.end(), odd_array.begin(), odd_array.end(), array.begin() + start);
}

void batcher_sort(std::vector<int>& array, int start, int end) {
  if (end - start <= 1) return;

  int mid = (start + end) / 2;

  batcher_sort(array, start, mid);
  batcher_sort(array, mid, end);

  batcher_merge(array, start, mid, end);
}

bool kovalchuk_a_odd_even::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  // Init array
  if (taskData->inputs_count[0] > 0) {
    array_ = std::vector<int>(taskData->inputs_count[0]);
    auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
    std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], array_.begin());
  } else {
    array_ = std::vector<int>();
  }
  // Init result vector
  result_ = std::vector<int>(taskData->inputs_count[0], 0);
  return true;
}

bool kovalchuk_a_odd_even::TestMPITaskSequential::validation() {
  internal_order_test();
  // Check count elements of output
  return taskData->outputs_count[0] == taskData->inputs_count[0];
}

bool kovalchuk_a_odd_even::TestMPITaskSequential::run() {
  internal_order_test();
  if (!array_.empty()) {
    batcher_sort(array_, 0, array_.size());
    result_ = array_;
  }
  return true;
}

bool kovalchuk_a_odd_even::TestMPITaskSequential::post_processing() {
  internal_order_test();
  std::copy(result_.begin(), result_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  return true;
}

bool kovalchuk_a_odd_even::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  // Init array to root
  if (world.rank() == 0) {
    int size = taskData->inputs_count[0];

    if (size > 0) {
      array_ = std::vector<int>(size);
      auto* tmp_ptr = reinterpret_cast<int*>(taskData->inputs[0]);
      std::copy(tmp_ptr, tmp_ptr + size, array_.begin());
    } else {
      array_ = std::vector<int>();
    }
  }

  return true;
}

bool kovalchuk_a_odd_even::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Check count elements of output
    return taskData->outputs_count[0] == taskData->inputs_count[0];
  }
  return true;
}

bool kovalchuk_a_odd_even::TestMPITaskParallel::run() {
  internal_order_test();

  int rank = world.rank();
  int size = world.size();

  int array_size = 0;
  if (world.rank() == 0) {
    array_size = taskData->inputs_count[0];
  }

  // Broadcast array size to all processes
  boost::mpi::broadcast(world, array_size, 0);

  int local_size = array_size / size;
  int extra_elements = array_size % size;

  std::vector<int> sendcounts(size, local_size);
  for (int i = 0; i < extra_elements; ++i) {
    sendcounts[i] += 1;
  }

  std::vector<int> displs(size, 0);
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + sendcounts[i - 1];
  }

  std::vector<int> empty_array(sendcounts[rank], 0);

  local_array_.resize(sendcounts[rank]);

  if (rank == 0) {
    boost::mpi::scatterv(world, array_.data(), sendcounts, displs, local_array_.data(), sendcounts[rank], 0);
  } else {
    boost::mpi::scatterv(world, empty_array.data(), sendcounts, displs, local_array_.data(), sendcounts[rank], 0);
  }

  batcher_sort(local_array_, 0, local_array_.size());

  result_.resize(array_size);
  boost::mpi::gatherv(world, local_array_.data(), sendcounts[rank], result_.data(), sendcounts, displs, 0);

  if (rank == 0) {
    batcher_sort(result_, 0, result_.size());
  }

  return true;
}

bool kovalchuk_a_odd_even::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    std::copy(result_.begin(), result_.end(), reinterpret_cast<int*>(taskData->outputs[0]));
  }
  return true;
}