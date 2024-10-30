// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gordeva_t_max_val_of_column_matrix/include/ops_mpi.hpp"

TEST(gordeva_t_max_val_of_column_matrix_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> max_s;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows = 5000, cols = 5000;

  if (world.rank() == 0) {
      global_matr =
 gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::gen_rand_matr(rows, cols);
    max_s.resize(cols, INT_MIN);
    for (auto& i : global_matr) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(i.data()));
    }
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_s.data()));
    taskDataPar->outputs_count.emplace_back(max_s.size());
  }

  // Create Perf attributes
  auto testMpiTaskParallel =
      std::make_shared<gordeva_t_max_val_of_column_matrix_mpi ::TestMPITaskParallel>(taskDataPar);

  // Create and init perf results
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf analyzer
  if (world.rank() == 0) {
    for (size_t j = 0; j < max_s.size(); ++j) {
      ASSERT_EQ(max_s[j], 200);
    }
  }
}

TEST(gordeva_t_max_val_of_column_matrix_mpi, test_task_run) {
  boost::mpi::communicator world;

  // Create TaskData
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> max_s;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int rows = 7000;
  int cols = 7000;

  if (world.rank() == 0) {
      global_matr = 
 gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskSequential::gen_rand_matr(rows, cols);
    max_s.resize(cols, INT_MIN);

    for (auto& i : global_matr) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(i.data()));
    }

    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(max_s.data()));
    taskDataPar->outputs_count.emplace_back(max_s.size());
  }

  // Create Perf attributes
  auto testMpiTaskParallel = std::make_shared<gordeva_t_max_val_of_column_matrix_mpi::TestMPITaskParallel>(taskDataPar);

  // Create and init perf results
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf analyzer
  if (world.rank() == 0) {
    for (size_t j = 0; j < max_s.size(); ++j) {
      ASSERT_EQ(max_s[j], 200);
    }
  }
}

// int main(int argc, char** argv) {
//   boost::mpi::environment env(argc, argv);
//   boost::mpi::communicator world;
//   ::testing::InitGoogleTest(&argc, argv);
//   ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
//   if (world.rank() != 0) {
//     delete listeners.Release(listeners.default_result_printer());
//   }
//   return RUN_ALL_TESTS();
// }
