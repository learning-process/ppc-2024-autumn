#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/laganina_e_sum_values_by_columns_matrix/include/ops_mpi.hpp"

TEST(laganina_e_sum_values_by_columns_matrix_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  int n = 3000;
  int m = 6000;

  // Create data
  std::vector<int> input(n * m, 0);
  std::vector<int> empty(n, 0);
  std::vector<int> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty.data()));
    taskDataPar->outputs_count.emplace_back(empty.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(empty, out);
  }
}

TEST(laganina_e_sum_values_by_columns_matrix_mpi, test_task_run) {
  boost::mpi::communicator world;
  int n = 3000;
  int m = 6000;

  // Create data
  std::vector<int> input(n * m, 0);
  std::vector<int> empty(n, 0);
  std::vector<int> out(n, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
    taskDataPar->inputs_count.emplace_back(input.size());
    taskDataPar->inputs_count.emplace_back(m);
    taskDataPar->inputs_count.emplace_back(n);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(empty.data()));
    taskDataPar->outputs_count.emplace_back(empty.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<laganina_e_sum_values_by_columns_matrix_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(empty, out);
  }
}