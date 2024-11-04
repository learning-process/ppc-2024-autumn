// Copyright 2023 Nasedkin Egor
#include <gtest/gtest.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>
#include "mpi/nasedkin_e_matrix_column_max_value/include/ops_mpi.hpp"

TEST(Parallel_Operations_MPI, Test_Matrix_Column_Max) {
  boost::mpi::communicator world;
  std::vector<int> global_matrix;
  std::vector<int> global_max(3, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int rows = 6, cols = 3;
    global_matrix = nasedkin_e_matrix_column_max_value_mpi::getRandomMatrix(rows, cols);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(global_matrix.size());
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }

  nasedkin_e_matrix_column_max_value_mpi::MatrixColumnMaxMPI matrixColumnMaxMPI(taskDataPar);
  ASSERT_EQ(matrixColumnMaxMPI.validation(), true);
  matrixColumnMaxMPI.pre_processing();
  matrixColumnMaxMPI.run();
  matrixColumnMaxMPI.post_processing();

  if (world.rank() == 0) {
    std::vector<int> reference_max(3, 0);
    for (int col = 0; col < 3; col++) {
      reference_max[col] = global_matrix[col];
      for (int row = 1; row < 6; row++) {
        if (global_matrix[row * 3 + col] > reference_max[col]) {
          reference_max[col] = global_matrix[row * 3 + col];
        }
      }
    }
    ASSERT_EQ(reference_max, global_max);
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}