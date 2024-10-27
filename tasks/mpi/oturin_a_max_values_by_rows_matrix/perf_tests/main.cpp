#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/oturin_a_max_values_by_rows_matrix/include/ops_mpi.hpp"

TEST(mpi_example_perf_test, test_pipeline_run) {
  size_t n = 300;
  size_t m = 300;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> global_max(m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_mat = std::vector<int>(n * m, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel>(taskDataPar, "max");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(n * m, global_max[0]);
  }
}

TEST(mpi_example_perf_test, test_task_run) {
  size_t n = 300;
  size_t m = 300;
  boost::mpi::communicator world;
  std::vector<int> global_mat;
  std::vector<int32_t> global_max(m, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_mat = std::vector<int>(n * m, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_mat.data()));
    taskDataPar->inputs_count.emplace_back(global_mat.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<oturin_a_max_values_by_rows_matrix_mpi::TestMPITaskParallel>(taskDataPar, "max");
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(n * m, global_max[0]);
  }
}
