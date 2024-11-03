// Copyright 2023 Nesterov Alexander
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
    // Initialization global_vector <T> with input data
    glob_v.resize(n);
    void* ptr_vec = glob_v.data();
    void* ptr_input = taskData->inputs[0];
    memcpy(ptr_vec, ptr_input, sizeof(T) * n);
  }
  try {
    boost::mpi::broadcast(world, n, 0);
  } catch (const boost::mpi::exception& e) {
    std::cerr << "Ошибка в MPI broadcast: " << e.what() << std::endl;
  }
  size_t scratter_length = n / size;  // minimum length to each process
  loc_v.resize(scratter_length);      // resize the local copy
  int scatterResult = MPI_Scatter(glob_v.data(), scratter_length * sizeof(T), MPI_BYTE, loc_v.data(),
                                  scratter_length * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
  if (scatterResult != MPI_SUCCESS) {
    char errorString[MPI_MAX_ERROR_STRING];
    int errorStringLength;
    MPI_Error_string(scatterResult, errorString, &errorStringLength);
    std::cerr << "Ошибка MPI_Scatter: " << errorString << std::endl;
  }
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::validation() {
  internal_order_test();
  // input && output data check
  if (rank == 0 && !taskData->inputs.empty() && !taskData->outputs.empty() && taskData->outputs_count[0] == 1) {
    return true;
  }
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::run() {
  internal_order_test();
  // counting violations locally
  count_num_of_orderly_violations_mpi();
  // redusing results
  try {
    boost::mpi::reduce(world, l_res, g_res, std::plus<unsigned long>(), 0);
  } catch (const boost::mpi::exception& e) {
    std::cerr << "Ошибка в MPI reduce: " << e.what() << std::endl;
    return 1;  // Завершение программы с кодом ошибки
  }
  if (rank == 0) {
    for (int i = 1; i < size; i++)  // are there any violations between local copies?
      if (glob_v[i * (n / size) - 1] > glob_v[i * (n / size)]) g_res++;
    for (size_t i = n - n % size; i < n; i++)  // are there any violations in the remainder?
      if (glob_v[i - 1] > glob_v[i]) g_res++;
  }
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::post_processing() {
  internal_order_test();
  // Pushing global result to output
  if (rank == 0) {
    reinterpret_cast<size_t*>(taskData->outputs[0])[0] = g_res;
  }
  return true;
}

template class kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<int>;
template class kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<double>;