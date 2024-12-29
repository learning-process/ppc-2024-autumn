// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kolodkin_g_image_contrast/include/ops_mpi.hpp"

TEST(mpi_kolodkin_g_image_contrast_test, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_image;
  std::vector<int32_t> global_out(999, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (unsigned long int i = 0; i < 999; i++) {
      global_image.push_back(0 + rand() % 255);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_image.data()));
    taskDataPar->inputs_count.emplace_back(global_image.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  auto testMpiTaskParallel = std::make_shared<kolodkin_g_image_contrast_mpi::TestMPITaskParallel>(taskDataPar);
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
  }
}

TEST(mpi_kolodkin_g_image_contrast_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_out(9999, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    for (unsigned long int i = 0; i < 9999; i++) {
      global_vec.push_back(0 + rand() % 255);
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(new std::vector<int>(global_out)));
  }

  auto testMpiTaskParallel = std::make_shared<kolodkin_g_image_contrast_mpi::TestMPITaskParallel>(taskDataPar);
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
  }
}