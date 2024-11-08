#include "mpi/shuravina_o_monte_carlo/include/ops_mpi.hpp"

#include <boost/mpi/collectives.hpp>
#include <random>

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel::pre_processing() {
  internal_order_test();
  integral_value_ = 0.0;
  return true;
}

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    // Дополнительные проверки для уверенности, что данные корректно инициализированы
    if (taskData->inputs_count.size() != 1 || taskData->outputs_count.size() != 1) {
      return false;
    }
    if (taskData->inputs_count[0] != 0 || taskData->outputs_count[0] != 1) {
      return false;
    }
    if (taskData->inputs[0] != nullptr || taskData->outputs[0] == nullptr) {
      return false;
    }
  }
  return true;
}

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel::run() {
  internal_order_test();
  int num_processes = world.size();
  int rank = world.rank();

  // Функция для интегрирования: f(x) = x^2
  auto f = [](double x) { return x * x; };

  // Интервал интегрирования [0, 1]
  double a = 0.0;
  double b = 1.0;

  // Количество точек для каждого процесса
  int num_points = 1000000;
  int local_num_points = num_points / num_processes;

  // Генерация случайных точек
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(a, b);

  double local_sum = 0.0;
  for (int i = 0; i < local_num_points; ++i) {
    double x = dis(gen);
    local_sum += f(x);
  }

  // Суммирование результатов от всех процессов
  double global_sum = 0.0;
  boost::mpi::reduce(world, local_sum, global_sum, std::plus<>(), 0);

  // Вычисление интеграла
  if (rank == 0) {
    integral_value_ = (global_sum / num_points) * (b - a);
  }

  return true;
}

bool shuravina_o_monte_carlo::MonteCarloIntegrationTaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<double*>(taskData->outputs[0])[0] = integral_value_;
  }
  return true;
}