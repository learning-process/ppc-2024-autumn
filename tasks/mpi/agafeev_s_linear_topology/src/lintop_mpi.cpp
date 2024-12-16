#include "mpi/agafeev_s_linear_topology/include/lintop_mpi.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>

namespace agafeev_s_linear_topology {

template <typename T>
bool LinearTopology<T>::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (taskData->outputs_count[0] == 1 && (taskData->inputs_count[0] > 0));
  }

  return true;
}

template <typename T>
bool LinearTopology<T>::pre_processing() {
  internal_order_test();

  if (world.size() == 1) return true;

  if (world.rank() == 0) {
    auto* temp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
    data_.insert(data_.begin(), temp_ptr, temp_ptr + taskData->inputs_count[0]);
  }

  return true;
}

template <typename T>
bool LinearTopology<T>::run() {
  internal_order_test();

  unsigned int w_rank = world.rank();
  unsigned int w_size = world.size();
  unsigned int data_size = 0;

  if (w_size == 1) return true;

  if (w_rank == 0) {
    data_size = taskData->inputs_count[0];
  }

  // data_.resize(data_size);

  boost::mpi::broadcast(world, data_size, 0);

  if (w_rank == 0) {
    ranks_vec_.push_back(0);
    world.send(1, 0, ranks_vec_);
    world.send(1, 1, data_);
  } else {
    world.recv(w_rank - 1, 0, ranks_vec_);
    world.recv(w_rank - 1, 1, data_);

    ranks_vec_.push_back(w_rank);

    if (w_rank != w_size - 1) {
      world.send(w_rank + 1, 0, ranks_vec_);
      world.send(w_rank + 1, 1, data_);
    }
  }

  if (w_rank == w_size - 1) {
    bool corr_order = std::is_sorted(ranks_vec_.begin(), ranks_vec_.end()) && ranks_vec_.size() == w_size;
    result_ = corr_order;
    world.send(0, 4, result_);
  }

  if (w_rank == 0) {
    world.recv(w_size - 1, 4, result_);
  }

  return true;
}

template <typename T>
bool LinearTopology<T>::post_processing() {
  internal_order_test();

  if (world.size() == 1) {
    result_ = true;
  }

  if (world.rank() == 0) {
    reinterpret_cast<bool*>(taskData->outputs[0])[0] = result_;
  }

  return true;
}

template class LinearTopology<int>;
template class LinearTopology<double>;

}  // namespace agafeev_s_linear_topology
