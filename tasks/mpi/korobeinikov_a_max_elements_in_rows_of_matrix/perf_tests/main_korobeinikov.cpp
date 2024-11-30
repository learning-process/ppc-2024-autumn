// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/korobeinikov_a_max_elements_in_rows_of_matrix/include/ops_mpi_korobeinikov.hpp"

TEST(mpi_korobeinikov_a_max_elements_in_rows_of_matrix_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  // Create data

  int count_rows = 100;  // not const, because reinterpret_cast does not work with const
  std::vector<int> global_matrix;
  std::vector<int> mpi_res(count_rows, 0);
  std::vector<int> right_answer(count_rows, 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix = std::vector<int>(count_rows * 500000, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpi_res.data()));
    taskDataPar->outputs_count.emplace_back(mpi_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<korobeinikov_a_test_task_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(mpi_res, right_answer);
  }
}

TEST(mpi_korobeinikov_a_max_elements_in_rows_of_matrix_perf_test, test_task_run) {
  boost::mpi::communicator world;

  // Create data

  int count_rows = 200;  // not const, because reinterpret_cast does not work with const
  std::vector<int> global_matrix;
  std::vector<int> mpi_res(count_rows, 0);
  std::vector<int> right_answer(count_rows, 1);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_matrix = std::vector<int>(count_rows * 500000, 1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(mpi_res.data()));
    taskDataPar->outputs_count.emplace_back(mpi_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<korobeinikov_a_test_task_mpi::TestMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(mpi_res, right_answer);
  }
}
