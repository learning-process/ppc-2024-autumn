// Golovkin Maksim
#pragma once

#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"

namespace golovkin_integration_rectangular_method {

class MPIIntegralCalculator : public ppc::core::Task {
 public:
  explicit MPIIntegralCalculator(std::shared_ptr<ppc::core::TaskData> taskData);

  bool validation() override;      // Проверка входных данных
  bool pre_processing() override;  // Предварительная обработка данных
  bool run() override;             // Выполнение интеграции с использованием MPI
  bool post_processing() override;  // Пост-обработка результатов и сборка итогового значения

 private:
  boost::mpi::communicator world;  
  std::shared_ptr<ppc::core::TaskData> taskData;
  double a, b, epsilon;  
  int cnt_of_splits;
  double h;
  double local_res;   // Локальный результат для каждого процесса
  double global_res;  // Глобальный результат, собираемый на процессе 0

  double function_square(double x);  // Функция для интегрирования
};

}  // namespace golovkin_integration_rectangular_method