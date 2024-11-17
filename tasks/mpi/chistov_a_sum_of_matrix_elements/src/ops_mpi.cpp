// Copyright 2023 Nesterov Alexander
#include "mpi/chistov_a_sum_of_matrix_elements/include/ops_mpi.hpp"

namespace chistov_a_sum_of_matrix_elements {

template <typename T>
bool TestMPITaskSequential<T>::pre_processing() {
  internal_order_test();

  T* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
  input_.assign(tmp_ptr, tmp_ptr + taskData->inputs_count[0]);
  return true;
}

template <typename T>
bool TestMPITaskSequential<T>::validation() {
  internal_order_test();
  return taskData->outputs_count[0] == 1;
}

template <typename T>
bool TestMPITaskSequential<T>::run() {
  internal_order_test();

  res = std::accumulate(input_.begin(), input_.end(), 0);
  return true;
}

template <typename T>
bool TestMPITaskSequential<T>::post_processing() {
  internal_order_test();

  reinterpret_cast<T*>(taskData->outputs[0])[0] = res;
  return true;
}

template <typename T>
bool TestMPITaskParallel<T>::pre_processing() {
  internal_order_test();

  int delta1 = 0;
  int delta2 = 0;

  if (world.rank() == 0) {
    n = static_cast<int>(taskData->inputs_count[1]);
    m = static_cast<int>(taskData->inputs_count[2]);
    int total_elements = n * m;
    delta1 = total_elements / world.size();
    delta2 = total_elements % world.size();
  }

  boost::mpi::broadcast(world, delta1, 0);
  boost::mpi::broadcast(world, delta2, 0);

  if (world.rank() == 0) {
    input_ = std::vector<T>(n * m);
    auto* tmp_ptr = reinterpret_cast<T*>(taskData->inputs[0]);
    for (int i = 0; i < static_cast<int>(taskData->inputs_count[0]); i++) {
      input_[i] = tmp_ptr[i];
    }

    int start_index = delta1 + (delta2 > 0 ? 1 : 0);
    for (int proc = 1; proc < world.size(); proc++) {
      int current_delta = delta1 + (proc < delta2 ? 1 : 0);
      world.send(proc, 0, input_.data() + start_index, current_delta);
      start_index += current_delta;
    }
  }

  int local_size = delta1 + (world.rank() < delta2 ? 1 : 0);
  local_input_ = std::vector<T>(local_size);

  if (world.rank() == 0) {
    local_input_ = std::vector<T>(input_.begin(), input_.begin() + local_size);
  } else {
    world.recv(0, 0, local_input_.data(), local_size);
  }

  return true;
}

template <typename T>
bool TestMPITaskParallel<T>::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    return (taskData->outputs_count[0] == 1 && !(taskData->inputs.empty()));
  }
  return true;
}

template <typename T>
bool TestMPITaskParallel<T>::run() {
  internal_order_test();

  T local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  reduce(world, local_res, res, std::plus<T>(), 0);

  return true;
}

template <typename T>
bool TestMPITaskParallel<T>::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<T*>(taskData->outputs[0])[0] = res;
  }
  return true;
}

template class TestMPITaskSequential<int>;
template class TestMPITaskSequential<double>;
template class TestMPITaskParallel<int>;
template class TestMPITaskParallel<double>;

}  // namespace chistov_a_sum_of_matrix_elements
