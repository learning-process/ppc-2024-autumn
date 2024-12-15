#include "mpi/agafeev_s_linear_topology/include/lintop_mpi.hpp"

namespace agafeev_s_linear_topology {

template <typename T>
bool LinearTopology<T>::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return taskData->inputs_count[0] > 0;
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

  unsigned int data_size = 0;
  if (world.rank() == 0) {
    data_size = taskData->inputs_count[0];
  }

  boost::mpi::broadcast(world, data_size, 0);

  return true;
}

template <typename T>
bool LinearTopology<T>::post_processing() {
  internal_order_test();

  if (world.rank() == 0) {
    reinterpret_cast<bool*>(taskData->outputs[0])[0] = result_;
  }

  return true;
}

}  // namespace agafeev_s_linear_topology
