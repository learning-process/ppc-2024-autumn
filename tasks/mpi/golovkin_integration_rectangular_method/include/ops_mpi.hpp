// Golovkin Maksim
#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace golovkin_integration_rectangular_method {

class MPIIntegralCalculator : public ppc::core::Task {
 public:
  explicit MPIIntegralCalculator(std::shared_ptr<ppc::core::TaskData> inputData) : Task(std::move(inputData)) {};

  bool validation() override;       // Проверка корректности входных данных
  bool pre_processing() override;   // Предобработка данных
  bool post_processing() override;  // Запуск интеграции с использованием MPI
  bool run() override;              // Постобработка и сбор итогового значения
  void set_function(const std::function<double(double)>& target_func);  // Задание функции для интеграции

 private:
  boost::mpi::communicator world;
  std::function<double(double)> function_;
  double lower_bound{};
  double upper_bound{};
  int num_partitions{};
  double global_result{};  // Итоговый результат, собираемый на главном процессе

  static double integrate(const std::function<double(double)>& f, double a, double b, int splits);
};

}  // namespace golovkin_integration_rectangular_method