#include "mpi/kovalev_k_num_of_orderly_violations/include/header.hpp"

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::count_num_of_orderly_violations_mpi() {
  for (size_t i = 1; i < loc_v.size(); i++) {
    if (loc_v[i - 1] > loc_v[i]) {
      l_res++;
    }
  }
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::pre_processing() {
  internal_order_test();
  g_res = l_res = 0;
  rank = world.rank();
  size = world.size();
  if (rank == 0) {
    n = taskData->inputs_count[0];
    glob_v.resize(n);
    void* ptr_vec = glob_v.data();
    void* ptr_input = taskData->inputs[0];
    memcpy(ptr_vec, ptr_input, sizeof(T) * n);
  }
  boost::mpi::broadcast(world, n, 0);
  int scratter_length = n / size;
  loc_v.resize(scratter_length);
  std::vector<int> sendcounts(size, scratter_length);
  std::vector<int> displs(size, 0);
  for (int i = 1; i < size; i++) displs[i] = displs[i] + scratter_length;
  boost::mpi::scatter(world, glob_v.data(), loc_v.data(), scratter_length, 0);
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    if (taskData->inputs.empty() || taskData->outputs.empty() || taskData->inputs_count[0] <= 0 ||
        taskData->outputs_count[0] != 1) {
      return false;
    }
  }
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::run() {
  internal_order_test();
  count_num_of_orderly_violations_mpi();
  boost::mpi::reduce(world, l_res, g_res, std::plus<unsigned long>(), 0);
  if (rank == 0) {
    for (int i = 1; i < size; i++)
      if (glob_v[i * (n / size) - 1] > glob_v[i * (n / size)]) g_res++;
    for (size_t i = n - n % size; i < n; i++)
      if (glob_v[i - 1] > glob_v[i]) g_res++;
  }
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::post_processing() {
  internal_order_test();
  if (rank == 0) {
    reinterpret_cast<size_t*>(taskData->outputs[0])[0] = g_res;
  }
  return true;
}

template class kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<int>;
template class kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<double>;