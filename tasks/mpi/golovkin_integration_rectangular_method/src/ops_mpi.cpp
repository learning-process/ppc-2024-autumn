//Golovkin Maksim
#include "mpi/golovkin_integration_rectangular_method/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace golovkin_integration_rectangular_method;

MPIIntegralCalculator::MPIIntegralCalculator(std::shared_ptr<ppc::core::TaskData> taskData)
    : ppc::core::Task(taskData), taskData(std::move(taskData)), local_res(0.0), global_res(0.0) {}

bool MPIIntegralCalculator::validation() { return (taskData->inputs.size() == 3 && taskData->outputs.size() >= 1); }

bool MPIIntegralCalculator::pre_processing() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    cnt_of_splits = *reinterpret_cast<int*>(taskData->inputs[2]); 
  }

  // Распространение значений на все процессы
  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cnt_of_splits, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (cnt_of_splits <= 0) return false;

  h = (b - a) / cnt_of_splits;

 // std::cout << "Process " << rank << " - a: " << a << ", b: " << b << ", cnt_of_splits: " << cnt_of_splits
            //<< ", h: " << h << std::endl;

  return true;
}

bool MPIIntegralCalculator::run() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Распределение разбиений
  int splits_per_proc = cnt_of_splits / size;
  int remaining_splits = cnt_of_splits % size;
  int start = rank * splits_per_proc + std::min(rank, remaining_splits);
  int end = start + splits_per_proc + (rank < remaining_splits ? 1 : 0);

  // Проверка правильности диапазона
  if (start >= end) {
    //std::cerr << "Process " << rank << " has no work to do." << std::endl;
    return false;  
  }

  // Вычисление локального результата
  double local_result = 0.0;
  for (int i = start; i < end; ++i) {
    double x = a + i * h;
    local_result += function_square(x);
  }
  local_res = local_result * h;

  //std::cout << "Process " << rank << " calculated local_res: " << local_res << std::endl;

  // Сбор результатов
  MPI_Reduce(&local_res, &global_res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    //std::cout << "Root process has global result after reduction: " << global_res << std::endl;
  }

  return true;
}

bool MPIIntegralCalculator::post_processing() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (taskData->outputs.empty()) return false;
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_res;  
    // Если есть дополнительные выходные данные
    // taskData->outputs[1] = <some other value>;
    // taskData->outputs[2] = <another value>;
  }

  // Рассылка результата всем процессам
  MPI_Bcast(&global_res, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank != 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_res;
  }

  return true;
}

double MPIIntegralCalculator::function_square(double x) {
  return x * x; 
}