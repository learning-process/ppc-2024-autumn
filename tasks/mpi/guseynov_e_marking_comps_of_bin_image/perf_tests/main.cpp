// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/guseynov_e_marking_comps_of_bin_image/include/ops_mpi.hpp"

TEST(guseynov_e_marking_comps_of_bin_image_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  const int rows = 1250;
  const int columns = 1250;
  std::vector<int> in(rows * columns);
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out(rows * columns);
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int pos = x * columns + y;
      if (x < 50) {
        in[pos] = 0;
        expected_out[pos] = 2;
      } else if (x == 50) {
        in[pos] = 1;
        expected_out[pos] = 1;
      } else if (x == 51) {
        in[pos] = 0;
        expected_out[pos] = 3;
      } else if (x == 52) {
        in[pos] = 1;
        expected_out[pos] = 1;
      } else {
        in[pos] = 0;
        expected_out[pos] = 4;
      }
    }
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(columns);
  }

  auto testMpiTaskParallel =
      std::make_shared<guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel->validation());
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
    ASSERT_EQ(out, expected_out);
  }
}

TEST(guseynov_e_marking_comps_of_bin_image_mpi, test_task_run) {
  boost::mpi::communicator world;
  const int rows = 1250;
  const int columns = 1250;
  std::vector<int> in(rows * columns);
  std::vector<int> out(rows * columns, -1);
  std::vector<int> expected_out(rows * columns);
  for (int x = 0; x < rows; x++) {
    for (int y = 0; y < columns; y++) {
      int pos = x * columns + y;
      if (x < 50) {
        in[pos] = 0;
        expected_out[pos] = 2;
      } else if (x == 50) {
        in[pos] = 1;
        expected_out[pos] = 1;
      } else if (x == 51) {
        in[pos] = 0;
        expected_out[pos] = 3;
      } else if (x == 52) {
        in[pos] = 1;
        expected_out[pos] = 1;
      } else {
        in[pos] = 0;
        expected_out[pos] = 4;
      }
    }
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(rows);
    taskDataPar->outputs_count.emplace_back(columns);
  }

  auto testMpiTaskParallel =
      std::make_shared<guseynov_e_marking_comps_of_bin_image_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(out, expected_out);
  }
}