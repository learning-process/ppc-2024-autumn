#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/smirnov_i_binary_segmentation/include/ops_mpi.hpp"

TEST(smirnov_i_binary_segmentation_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int cols = 2011;
  int rows = 1193;
  std::vector<int> img;
  std::vector<int> mask;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    img = std::vector<int>(cols * rows, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);

    mask = std::vector<int>(cols * rows, 1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
    taskDataPar->outputs_count.emplace_back(cols);
    taskDataPar->outputs_count.emplace_back(rows);
  }

  auto testMpiTaskParallel = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskParallel>(taskDataPar);
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
    std::vector<int> expected_mask(cols * rows, 2);

    for (int i = 0; i < cols * rows; i++) {
      ASSERT_EQ(expected_mask[i], mask[i]);
    }
  }
}

TEST(smirnov_i_binary_segmentation_mpi, test_task_run) {
  boost::mpi::communicator world;
  int cols = 2011;
  int rows = 1193;
  std::vector<int> img;
  std::vector<int> mask;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    img = std::vector<int>(cols * rows, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(img.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);

    mask = std::vector<int>(cols * rows, 1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(mask.data()));
    taskDataPar->outputs_count.emplace_back(cols);
    taskDataPar->outputs_count.emplace_back(rows);
  }

  auto testMpiTaskParallel = std::make_shared<smirnov_i_binary_segmentation::TestMPITaskParallel>(taskDataPar);
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
    std::vector<int> expected_mask(cols * rows, 2);

    for (int i = 0; i < cols * rows; i++) {
      ASSERT_EQ(expected_mask[i], mask[i]);
    }
  }
}