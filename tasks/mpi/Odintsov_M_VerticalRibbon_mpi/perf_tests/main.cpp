
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/Odintsov_M_VerticalRibbon_mpi/include/ops_mpi.hpp"

TEST(Odincov_M_VerticalRibbon_MPI_parallel_perf_test, my_test_pipeline_run) {
  // Create data
  boost::mpi::communicator com;

  // Create data
  std::vector<double> matrixA(14400, 1);
  std::vector<double> vectorB(120, 1);
  std::vector<double> out(120, 0);
  std::vector<double> out_s(120, 120);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
    taskDataPar->inputs_count.emplace_back(14400);
    taskDataPar->inputs_count.emplace_back(120);
    taskDataPar->inputs_count.emplace_back(120);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(120);
  }
  // Create Task
  auto testClassPar = std::make_shared<Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPIParallel>(taskDataPar);
  ASSERT_TRUE(testClassPar->validation());
  testClassPar->pre_processing();
  testClassPar->run();
  testClassPar->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClassPar);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (com.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_EQ(out_s, out);
  }
}
TEST(Odincov_M_VerticalRibbon_MPI_parallel_perf_test, my_test_task_run) {
  // Create data
  boost::mpi::communicator com;

  // Create data
  std::vector<double> matrixA(14400, 1);
  std::vector<double> vectorB(120, 1);
  std::vector<double> out(120, 0);
  std::vector<double> out_s(120, 120);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
    taskDataPar->inputs_count.emplace_back(14400);
    taskDataPar->inputs_count.emplace_back(120);
    taskDataPar->inputs_count.emplace_back(120);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(120);
  }
  // Create Task
  auto testClassPar = std::make_shared<Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPIParallel>(taskDataPar);
  ASSERT_EQ(testClassPar->validation(), true);
  testClassPar->pre_processing();
  testClassPar->run();
  testClassPar->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 100;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClassPar);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (com.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_EQ(out_s, out);
  }
}