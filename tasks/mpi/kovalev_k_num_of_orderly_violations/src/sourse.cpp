// Copyright 2023 Nesterov Alexander
#include "mpi/kovalev_k_num_of_orderly_violations/include/header.hpp"

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::count_num_of_orderly_violations_mpi() {
  size_t local_n = n / size;
  size_t start_index = rank * local_n;
  size_t end_index = (rank == size - 1) ? n : start_index + local_n;
  for (size_t i = start_index + 1; i < end_index; i++) {
    if (v[i - 1] > v[i]) {
      l_res++;
    }
  }
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::pre_processing() {
  internal_order_test();
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  // Инициализация вектора<T> размером n
  v.resize(n);
  void* ptr_input = taskData->inputs[0];
  void* ptr_vec = v.data();
  MPI_Bcast(ptr_input, n * sizeof(T), MPI_BYTE, 0, MPI_COMM_WORLD);
  memcpy(ptr_vec, ptr_input, sizeof(T) * n);
  // Инициализация счетчика
  l_res = 0;
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::validation() {
  internal_order_test();
  // Проверка количества элементов на выходе
  return (taskData->inputs_count[0] == n && taskData->outputs_count[0] == 1);
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::run() {
  internal_order_test();
  count_num_of_orderly_violations_mpi();
  // Объединение локальных результатов в глобальный результат
  MPI_Reduce(&l_res, &g_res, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  return true;
}

template <class T>
bool kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<T>::post_processing() {
  internal_order_test();

  if (rank == 0) {
    reinterpret_cast<size_t*>(taskData->outputs[0])[0] = global_res;
  }
  return true;
}

template class kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<int>;
template class kovalev_k_num_of_orderly_violations_mpi::NumOfOrderlyViolationsPar<double>;