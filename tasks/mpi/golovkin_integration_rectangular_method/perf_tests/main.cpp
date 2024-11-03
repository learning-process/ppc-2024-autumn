//Golovkin Maksim

#include <gtest/gtest.h>
#include <string>
#include <vector>
#include <boost/mpi/timer.hpp>


#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi/golovkin_integration_rectangular_method/include/ops_mpi.hpp"



using namespace golovkin_integration_rectangular_method;
using ppc::core::Perf;
using ppc::core::TaskData;

TEST(golovkin_integration_rectangular_method, test_pipeline_run) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = 1.0;
  int n = 1000000;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  // Только процесс 0 инициализирует данные и делится ими с другими
  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.push_back(1);
  }


  auto testMpiTaskParallel =
      std::make_shared<golovkin_integration_rectangular_method::MPIIntegralCalculator>(taskDataPar);

  // Проверка валидности данных на каждом процессе
  ASSERT_TRUE(testMpiTaskParallel->validation());

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    double exact = 1.0 / 3.0;
    EXPECT_NEAR(output, exact, 1e-4);
    ppc::core::Perf::print_perf_statistic(perfResults);
  }

  double exact = 1.0 / 3.0;
  EXPECT_NEAR(output, exact, 1e-4);
}
TEST(golovkin_integration_rectangular_method, test_task_run) {
  boost::mpi::communicator world;
  double a = 0.0;
  double b = 1.0;
  int n = 1000000;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&a));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&b));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t*>(&n));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t*>(&output));
    taskDataPar->outputs_count.push_back(1);
  }

  auto testMpiTaskParallel =
      std::make_shared<golovkin_integration_rectangular_method::MPIIntegralCalculator>(taskDataPar);

  ASSERT_TRUE(testMpiTaskParallel->validation());

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    double exact = 1.0 / 3.0;
    EXPECT_NEAR(output, exact, 1e-4);
    ppc::core::Perf::print_perf_statistic(perfResults);
  }

  double exact = 1.0 / 3.0;
  EXPECT_NEAR(output, exact, 1e-4);
}