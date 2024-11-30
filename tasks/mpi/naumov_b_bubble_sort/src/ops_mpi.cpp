#include "mpi/naumov_b_bubble_sort/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <iostream>
#include <vector>

namespace naumov_b_bubble_sort_mpi {

std::vector<int> out;

bool TestMPITaskParallel::pre_processing() {
  int rank = world.rank();
  int size = world.size();

  // Инициализация данных на процессе 0
  if (rank == 0) {
    auto* input_ = reinterpret_cast<int*>(taskData->inputs[0]);  // Примерные данные
    std::cout << "Input size: " << input_.size() << std::endl;
    std::cout << "Input data (first 10 elements): ";
    for (int i = 0; i < input_.size(); ++i) {
      std::cout << input_[i] << " ";
    }
    std::cout << std::endl;
  }

  int total_size = input_.size();
  int quotient = total_size / size;
  int remainder = total_size % size;

  std::vector<int> send_counts(size, quotient);
  for (int i = 0; i < remainder; ++i) {
    send_counts[size - 1 - i]++;
  }

  std::vector<int> displs(size, 0);
  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + send_counts[i - 1];
  }

  

  int local_size = send_counts[rank];
  local_input_.resize(local_size);

  // Логируем данные на процессе 0
  if (rank == 0) {
    std::cout << "Process 0 sending data: ";
    for (int i = 0; i < input_.size(); ++i) {
      std::cout << input_[i] << " ";
    }
    std::cout << std::endl;
  }

  // Передаем данные с использованием MPI_Scatterv
  MPI_Scatterv(input_.data(), send_counts.data(), displs.data(), MPI_INT, local_input_.data(), local_size, MPI_INT, 0,
               MPI_COMM_WORLD);

  // Логируем данные после получения
  std::cout << "Process " << rank << " received local data: ";
  for (int i = 0; i < local_input_.size(); ++i) {
    std::cout << local_input_[i] << " ";
  }
  std::cout << std::endl;

  return true;
}

bool TestMPITaskParallel::validation() {
  internal_order_test();
  return true;
}

bool TestMPITaskParallel::run() {
  internal_order_test();
  int rank = world.rank();  // Текущий процесс

  // Логирование данных перед сортировкой
  std::cout << "Process " << rank << " before sorting: ";
  for (int i = 0; i < local_input_.size(); ++i) {
    std::cout << local_input_[i] << " ";
  }
  std::cout << std::endl;

  // Сортировка локальной части данных
  std::sort(local_input_.begin(), local_input_.end());

  // Логирование после сортировки
  std::cout << "Process " << rank << " sorted its local data: ";
  for (int i = 0; i < local_input_.size(); ++i) {
    std::cout << local_input_[i] << " ";
  }
  std::cout << std::endl;

  return true;
}

bool TestMPITaskParallel::post_processing() {
  internal_order_test();
  int rank = world.rank();
  int size = world.size();

  std::vector<int> gathered_data;
  std::vector<int> send_counts(size, 0);
  std::vector<int> displs(size, 0);

  int total_size = input_.size();
  int quotient = total_size / size;
  int remainder = total_size % size;

  for (int i = 0; i < size; ++i) {
    send_counts[i] = quotient + (i < remainder ? 1 : 0);
  }

  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i - 1] + send_counts[i - 1];
  }

  if (rank == 0) {
    gathered_data.resize(total_size);
  }

  // Сбор данных с помощью MPI_Gatherv
  MPI_Gatherv(local_input_.data(), local_input_.size(), MPI_INT, gathered_data.data(), send_counts.data(),
              displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    // Логируем собранные данные
    std::cout << "Gathered data on root: ";
    for (int i = 0; i < gathered_data.size(); ++i) {
      std::cout << gathered_data[i] << " ";
    }
    std::cout << std::endl;

    // Логируем перед финальной сортировкой
    std::cout << "Process " << rank << " before final sort: ";
    for (int i = 0; i < gathered_data.size(); ++i) {
      std::cout << gathered_data[i] << " ";
    }
    std::cout << std::endl;

    // Финальная сортировка
    std::sort(gathered_data.begin(), gathered_data.end());

    // Логируем после финальной сортировки
    std::cout << "Final sorted array on process 0: ";
    for (int i = 0; i < gathered_data.size(); ++i) {
      std::cout << gathered_data[i] << " ";
    }
    std::cout << std::endl;

    out = gathered_data;  // Передаем отсортированные данные в 'out'
  }

  // Логирование финальных данных в 'out'
  if (rank == 0) {
    std::cout << "Final output in 'out': ";
    for (int i = 0; i < out.size(); ++i) {
      std::cout << out[i] << " ";
    }
    std::cout << std::endl;
  }

  return true;
}

}  // namespace naumov_b_bubble_sort_mpi
