// Golovkin Maksims
#include "mpi/golovkin_integration_rectangular_method/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi.hpp>
#include <chrono>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace golovkin_integration_rectangular_method;
using namespace std::chrono_literals;

bool MPIIntegralCalculator::validation() {
  internal_order_test();

  // Начало отсчета времени выполнения
  auto start = std::chrono::high_resolution_clock::now();
  int timeout_ms = 3000;  // Устанавливаем тайм-аут для валидации (например, 3000 миллисекунд)

  bool is_valid = true;

  if (world.rank() == 0 || world.rank() == 1 || world.rank() == 2 || world.rank() == 3) {
    is_valid = taskData->inputs_count[0] > 0 && taskData->inputs_count[1] > 0 && taskData->outputs_count[0] == 1;

    // Лог для отладки
    if (!is_valid) {
      std::cerr << "Validation failed on rank 0 with inputs_count or outputs_count invalid\n";
    }
  }

  // Синхронизация и широковещательная передача результата валидации другим процессам
  broadcast(world, is_valid, 0);

  // Проверка на тайм-аут
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  if (duration.count() > timeout_ms) {
    std::cerr << "Timeout in validation on rank " << world.rank() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  return is_valid;
}
bool MPIIntegralCalculator::pre_processing() {
  internal_order_test();

  // Начало отсчета времени выполнения
  auto start = std::chrono::high_resolution_clock::now();
  int timeout_ms = 5000;  // Максимальное время выполнения в миллисекундах

  if (world.rank() == 0 || world.rank() == 1 || world.rank() == 2 || world.rank() == 3) {
    auto* start_ptr = reinterpret_cast<double*>(taskData->inputs[0]);
    auto* end_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
    auto* split_ptr = reinterpret_cast<int*>(taskData->inputs[2]);

    lower_bound = *start_ptr;
    upper_bound = *end_ptr;
    num_partitions = *split_ptr;
  }

  broadcast(world, lower_bound, 0);
  broadcast(world, upper_bound, 0);
  broadcast(world, num_partitions, 0);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  if (duration.count() > timeout_ms) {
    std::cerr << "Timeout in pre_processing on rank " << world.rank() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  return true;
}

bool MPIIntegralCalculator::run() {
  internal_order_test();

  auto start = std::chrono::high_resolution_clock::now();
  int timeout_ms = 10000;

  double local_result{};
  local_result = integrate(function_, lower_bound, upper_bound, num_partitions);

  reduce(world, local_result, global_result, std::plus<>(), 0);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  if (duration.count() > timeout_ms) {
    std::cerr << "Timeout in run on rank " << world.rank() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  return true;
}

bool MPIIntegralCalculator::post_processing() {
  internal_order_test();

  auto start = std::chrono::high_resolution_clock::now();
  int timeout_ms = 5000;

  if (world.rank() == 0 || world.rank() == 1 || world.rank() == 2 || world.rank() == 3) {
    *reinterpret_cast<double*>(taskData->outputs[0]) = global_result;
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  if (duration.count() > timeout_ms) {
    std::cerr << "Timeout in post_processing on rank " << world.rank() << "\n";
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  return true;
}

double MPIIntegralCalculator::integrate(const std::function<double(double)>& f, double a, double b, int splits) {
  int current_process = world.rank();
  int total_processes = world.size();
  double step_size;
  double local_sum = 0.0;
  step_size = (b - a) / splits;

  auto start = std::chrono::high_resolution_clock::now();
  int timeout_ms = 5000;

  for (int i = current_process; i < splits; i += total_processes) {
    double x = a + i * step_size;
    local_sum += f(x) * step_size;

    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);
    if (duration.count() > timeout_ms) {
      std::cerr << "Timeout in integrate on rank " << current_process << "\n";
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  return local_sum;
}

void MPIIntegralCalculator::set_function(const std::function<double(double)>& target_func) { function_ = target_func; }