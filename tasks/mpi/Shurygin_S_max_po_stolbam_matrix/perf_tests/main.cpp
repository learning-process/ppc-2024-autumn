// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/Shurygin_S_max_po_stolbam_matrix/include/ops_mpi.hpp"

TEST(Shurygin_S_max_po_stolbam_matrix_mpi_perf_test, test_pipeline_run_max) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_rows = 5000;
  int count_columns = 5000;
  if (world.rank() == 0) {
    global_matrix =
        Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::generate_random_matrix(count_rows, count_columns);
    global_max.resize(count_columns, INT_MIN);
    for (auto& row : global_matrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  auto testMpiTaskParallel = std::make_shared<Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  if (world.rank() == 0) {
    for (size_t j = 0; j < global_max.size(); ++j) {
      int max_val = global_matrix[0][j];
      for (int i = 1; i < count_rows; ++i) {
        if (global_matrix[i][j] > max_val) {
          max_val = global_matrix[i][j];
        }
      }
      ASSERT_EQ(global_max[j], max_val);
    }
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_mpi_perf_test, test_task_run_max) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matrix;
  std::vector<int32_t> global_max;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_rows = 5000;
  int count_columns = 5000;
  if (world.rank() == 0) {
    global_matrix =
        Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskSequential::generate_random_matrix(count_rows, count_columns);
    global_max.resize(count_columns, INT_MIN);
    for (auto& row : global_matrix) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }
    taskDataPar->inputs_count.emplace_back(count_rows);
    taskDataPar->inputs_count.emplace_back(count_columns);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  auto testMpiTaskParallel = std::make_shared<Shurygin_S_max_po_stolbam_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  if (world.rank() == 0) {
    for (size_t j = 0; j < global_max.size(); ++j) {
      int max_val = global_matrix[0][j];
      for (int i = 1; i < count_rows; ++i) {
        if (global_matrix[i][j] > max_val) {
          max_val = global_matrix[i][j];
        }
      }
      ASSERT_EQ(global_max[j], max_val);
    }
  }
}