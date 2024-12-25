#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/plekhanov_d_verticalgaus/include/ops_mpi.hpp"

TEST(plekhanov_d_verticalgaus_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int width = 3000;
  const int height = 3000;

  std::vector<double> global_img(width * height, 1);
  std::vector<double> global_ans(width * height, 0);
  std::vector<double> ans(width * height, 1);

  for (int i = 0; i < width; ++i) {
    ans[i] = 0;
    ans[(height - 1) * width + i] = 0;
  }
  for (int i = 1; i < height - 1; ++i) {
    ans[i * width] = 0.75;
    ans[i * width + width - 1] = 0.75;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<plekhanov_d_verticalgaus_mpi::VerticalGausMPITest>(taskDataPar);
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
    ASSERT_EQ(global_ans, ans);
  }
}

TEST(plekhanov_d_verticalgaus_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int width = 3000;
  const int height = 3000;

  std::vector<double> global_img(width * height, 1);
  std::vector<double> global_ans(width * height, 0);
  std::vector<double> ans(width * height, 1);

  for (int i = 0; i < width; ++i) {
    ans[i] = 0;
    ans[(height - 1) * width + i] = 0;
  }
  for (int i = 1; i < height - 1; ++i) {
    ans[i * width] = 0.75;
    ans[i * width + width - 1] = 0.75;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_img.data()));
    taskDataPar->inputs_count.emplace_back(global_img.size());
    taskDataPar->inputs_count.emplace_back(width);
    taskDataPar->inputs_count.emplace_back(height);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_ans.data()));
    taskDataPar->outputs_count.emplace_back(global_ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<plekhanov_d_verticalgaus_mpi::VerticalGausMPITest>(taskDataPar);
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
    ASSERT_EQ(global_ans, ans);
  }
}