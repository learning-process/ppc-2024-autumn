// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/korovin_n_min_val_row_matrix/include/ops_mpi.hpp"

TEST(korovin_n_min_val_row_matrix_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_rows;
  int count_columns;

  if (world.rank() == 0) {
    count_rows = 3000;
    count_columns = 3000;
    global_matrix =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(count_rows, count_columns);
    global_min.resize(count_rows, INT_MAX);

    for (auto& row : global_matrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto testMpiTaskParallel = std::make_shared<korovin_n_min_val_row_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (size_t i = 0; i < global_min.size(); ++i) {
      ASSERT_EQ(global_min[i], INT_MIN);
    }
  }
}

TEST(korovin_n_min_val_row_matrix_mpi_perf_test, test_task_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_rows;
  int count_columns;

  if (world.rank() == 0) {
    count_rows = 3000;
    count_columns = 3000;
    global_matrix =
        korovin_n_min_val_row_matrix_mpi::TestMPITaskSequential::generate_rnd_matrix(count_rows, count_columns);
    global_min.resize(count_rows, INT_MAX);

    for (auto& row : global_matrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto testMpiTaskParallel = std::make_shared<korovin_n_min_val_row_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (size_t i = 0; i < global_min.size(); ++i) {
      ASSERT_EQ(global_min[i], INT_MIN);
    }
  }
}
