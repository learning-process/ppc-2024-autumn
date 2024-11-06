// Golovkin Maksim
#include "mpi/golovkin_integration_rectangular_method/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace golovkin_integration_rectangular_method;

MPIIntegralCalculator::MPIIntegralCalculator(std::shared_ptr<ppc::core::TaskData> taskData)
    : ppc::core::Task(taskData), taskData(std::move(taskData)), local_res(0.0), global_res(0.0) {}

bool MPIIntegralCalculator::validation() {
  if (taskData->inputs.empty() || taskData->outputs.empty()) {
    return false;
  }
  // Убедимся, что только процесс 0 может инициировать данные
  if (taskData->inputs.size() < 3 || (taskData->outputs.size() < 1 && world.rank() == 0)) {
    return false;
  }
  return true;
}

bool MPIIntegralCalculator::pre_processing() {
  if (world.rank() == 0) {
    a = *reinterpret_cast<double*>(taskData->inputs[0]);
    b = *reinterpret_cast<double*>(taskData->inputs[1]);
    cnt_of_splits = *reinterpret_cast<int*>(taskData->inputs[2]);
  }

  // Распространение значений на все процессы
  MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&cnt_of_splits, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Проверка корректности количества разбиений
  if (cnt_of_splits <= 0) return false;

  h = (b - a) / cnt_of_splits;  // Вычисление ширины подынтервала
    std::cout << "Process " << world.rank() << " - a: " << a << ", b: " << b << ", cnt_of_splits: " << cnt_of_splits<< ", h: " << h << std::endl;

  return true;
}

bool MPIIntegralCalculator::run() {
  // Проверка, что cnt_of_splits, a, и h инициализированы и имеют корректные значения
  if (cnt_of_splits <= 0 || h <= 0.0 || a >= b) {
    std::cerr << "Process " << world.rank() << ": Invalid configuration (cnt_of_splits, h, or range a-b)" << std::endl;
    return false;
  }
  std::cout << "Process ";
  // Делим работу между процессами
  int splits_per_proc = cnt_of_splits / world.size();
  int remaining_splits = cnt_of_splits % world.size();
  int start = world.rank() * splits_per_proc + std::min(world.rank(), remaining_splits);
  int end = start + splits_per_proc + (world.rank() < remaining_splits ? 1 : 0);
  std::cout << "Process1 ";
  // Проверка диапазона
  if (start >= end) {
    std::cerr << "Process " << world.rank() << " has no work to do (start >= end)." << std::endl;
    local_res = 0.0;  // Устанавливаем local_res для процесса без работы
  } else {
    // Вычисление локального результата
    double local_result = 0.0;
    for (int i = start; i < end; ++i) {
      double x = a + i * h;
      local_result += function_square(x);  // Функция, которую мы интегрируем
    }
    local_res = local_result * h;  // Умножаем на ширину подынтервала
  }
  std::cout << "Process2 ";
  // Сбор результатов, проверка глобальной синхронизации
  double local_global_res = 0.0;
  MPI_Reduce(&local_res, &local_global_res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  std::cout << "Process3 ";
  if (world.rank() == 0) {
    global_res = local_global_res;
    std::cout << "Root process has global result after reduction: " << global_res << std::endl;
  }
  std::cout << "Process4 ";
  return true;
}

bool MPIIntegralCalculator::post_processing() {
  if (world.rank() == 0) {
    if (taskData->outputs.empty()) return false;
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_res;
  }

  if (world.rank() != 0) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_res;
  }

  return true;
}

double MPIIntegralCalculator::function_square(double x) {
  return x * x;  // Пример функции, которую мы интегрируем
}