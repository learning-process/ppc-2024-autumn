// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/korovin_n_min_val_row_matrix/include/ops_mpi.hpp"

TEST(korovin_n_min_val_row_matrix_mpi, test_pipeline_run_min) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_rows, count_columns;

  if (world.rank() == 0) {
    count_rows = 5000;
    count_columns = 5000;
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

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_min.size(); ++i) {
      ASSERT_EQ(global_min[i], -25);
    }
  }
}

TEST(korovin_n_min_val_row_matrix_mpi_perf_test, test_task_run_min) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_min;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_rows, count_columns;

  if (world.rank() == 0) {
    count_rows = 5000;
    count_columns = 5000;
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

  if (world.rank() == 0) {
    for (size_t i = 0; i < global_min.size(); ++i) {
      ASSERT_EQ(global_min[i], -25);
    }
  }
}
