// Golovkin Maksim

#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <string>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/golovkin_integration_rectangular_method/include/ops_mpi.hpp"

using namespace golovkin_integration_rectangular_method;
using ppc::core::Perf;
using ppc::core::TaskData;

TEST(golovkin_integration_rectangular_method, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = 0.0;
  double b = 5.0;
  int cnt_of_splits = 1000000;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cnt_of_splits));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<golovkin_integration_rectangular_method::MPIIntegralCalculator>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return x + 2.0; });
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Создание атрибутов Perf
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создание и инициализация результатов производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание анализатора производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double expected_value = (b * (b + 4.0) / 2.0) - (a * (a + 4.0) / 2.0);  // Аналитическое значение интеграла
    ASSERT_NEAR(global_result[0], expected_value, 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> global_result(1, 0);

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double a = 1.0;
  double b = 10.0;
  int cnt_of_splits = 1000000;
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&cnt_of_splits));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<golovkin_integration_rectangular_method::MPIIntegralCalculator>(taskDataPar);
  testMpiTaskParallel->set_function([](double x) { return std::log(x); });
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Создание атрибутов Perf
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Создание и инициализация результатов производительности
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Создание анализатора производительности
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    double expected_value = b * std::log(b) - b - (a * std::log(a) - a);  // Аналитическое значение интеграла ln(x)
    ASSERT_NEAR(global_result[0], expected_value, 1e-3);
  }
}