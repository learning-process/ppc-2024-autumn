// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/morozov_e_mult_sparse_matrix/include/ops_mpi.hpp"
TEST(morozov_e_mult_sparse_matrix, Test_Validation_colsA_notEqual_rowsB) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrixA = {{0, 2}, {1, 0}, {0, 4}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(matrixA.size());
    taskData->inputs_count.emplace_back(matrixA[0].size());
    taskData->inputs_count.emplace_back(dA.size());
    taskData->inputs_count.emplace_back(row_indA.size());
    taskData->inputs_count.emplace_back(col_indA.size());

    taskData->inputs_count.emplace_back(matrixB.size());
    taskData->inputs_count.emplace_back(matrixB[0].size());
    taskData->inputs_count.emplace_back(dB.size());
    taskData->inputs_count.emplace_back(row_indB.size());
    taskData->inputs_count.emplace_back(col_indB.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < out.size(); ++i) {
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    }
    taskData->outputs_count.emplace_back(out.size());
    taskData->outputs_count.emplace_back(out[0].size());
  }

  morozov_e_mult_sparse_matrix::TestMPITaskParallel testMpiTaskParallel(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}
TEST(morozov_e_mult_sparse_matrix, Test_Validation_colsAns_notEqual_colsB) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrixA = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size() + 1, 0));
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(matrixA.size());
    taskData->inputs_count.emplace_back(matrixA[0].size());
    taskData->inputs_count.emplace_back(dA.size());
    taskData->inputs_count.emplace_back(row_indA.size());
    taskData->inputs_count.emplace_back(col_indA.size());

    taskData->inputs_count.emplace_back(matrixB.size());
    taskData->inputs_count.emplace_back(matrixB[0].size());
    taskData->inputs_count.emplace_back(dB.size());
    taskData->inputs_count.emplace_back(row_indB.size());
    taskData->inputs_count.emplace_back(col_indB.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < out.size(); ++i) {
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    }
    taskData->outputs_count.emplace_back(out.size());
    taskData->outputs_count.emplace_back(out[0].size());
  }

  morozov_e_mult_sparse_matrix::TestMPITaskParallel testMpiTaskParallel(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}
TEST(morozov_e_mult_sparse_matrix, Test_Validation_rowsAns_notEqual_rowsA) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrixA = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out(matrixA.size() + 2, std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(matrixA.size());
    taskData->inputs_count.emplace_back(matrixA[0].size());
    taskData->inputs_count.emplace_back(dA.size());
    taskData->inputs_count.emplace_back(row_indA.size());
    taskData->inputs_count.emplace_back(col_indA.size());

    taskData->inputs_count.emplace_back(matrixB.size());
    taskData->inputs_count.emplace_back(matrixB[0].size());
    taskData->inputs_count.emplace_back(dB.size());
    taskData->inputs_count.emplace_back(row_indB.size());
    taskData->inputs_count.emplace_back(col_indB.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < out.size(); ++i) {
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    }
    taskData->outputs_count.emplace_back(out.size());
    taskData->outputs_count.emplace_back(out[0].size());
  }

  morozov_e_mult_sparse_matrix::TestMPITaskParallel testMpiTaskParallel(taskData);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}
TEST(morozov_e_mult_sparse_matrix, Test_Main1) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrixA = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(matrixA.size());
    taskData->inputs_count.emplace_back(matrixA[0].size());
    taskData->inputs_count.emplace_back(dA.size());
    taskData->inputs_count.emplace_back(row_indA.size());
    taskData->inputs_count.emplace_back(col_indA.size());

    taskData->inputs_count.emplace_back(matrixB.size());
    taskData->inputs_count.emplace_back(matrixB[0].size());
    taskData->inputs_count.emplace_back(dB.size());
    taskData->inputs_count.emplace_back(row_indB.size());
    taskData->inputs_count.emplace_back(col_indB.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < out.size(); ++i) {
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    }
    taskData->outputs_count.emplace_back(out.size());
    taskData->outputs_count.emplace_back(out[0].size());
  }

  morozov_e_mult_sparse_matrix::TestMPITaskParallel testMpiTaskParallel(taskData);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::vector<double>> ans(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    for (size_t i = 0; i < out.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskData->outputs[i]);
      ans[i] = std::vector(ptr, ptr + matrixB.size());
    }
    std::vector<std::vector<double>> check_result = {{2, 0, 6}, {0, 14, 0}, {4, 0, 12}};
    ASSERT_EQ(check_result, ans);
  }
}
TEST(morozov_e_mult_sparse_matrix, Test_Main2) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrixA = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  std::vector<std::vector<double>> matrixB = {{2, 0}, {0, 3}, {10, 4}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(matrixA.size());
    taskData->inputs_count.emplace_back(matrixA[0].size());
    taskData->inputs_count.emplace_back(dA.size());
    taskData->inputs_count.emplace_back(row_indA.size());
    taskData->inputs_count.emplace_back(col_indA.size());

    taskData->inputs_count.emplace_back(matrixB.size());
    taskData->inputs_count.emplace_back(matrixB[0].size());
    taskData->inputs_count.emplace_back(dB.size());
    taskData->inputs_count.emplace_back(row_indB.size());
    taskData->inputs_count.emplace_back(col_indB.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < out.size(); ++i) {
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    }
    taskData->outputs_count.emplace_back(out.size());
    taskData->outputs_count.emplace_back(out[0].size());
  }

  morozov_e_mult_sparse_matrix::TestMPITaskParallel testMpiTaskParallel(taskData);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::vector<double>> ans(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    for (size_t i = 0; i < out.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskData->outputs[i]);
      ans[i] = std::vector(ptr, ptr + matrixB[0].size());
    }
    ASSERT_EQ(matrixB, ans);
  }
}
TEST(morozov_e_mult_sparse_matrix, Test_Main3) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrixA = {{2, 0, 5}, {0, 4, 6}};
  std::vector<std::vector<double>> matrixB = {{2, 0, 5}, {0, 3, 1}, {5, 0, 12}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    taskData->inputs_count.emplace_back(matrixA.size());
    taskData->inputs_count.emplace_back(matrixA[0].size());
    taskData->inputs_count.emplace_back(dA.size());
    taskData->inputs_count.emplace_back(row_indA.size());
    taskData->inputs_count.emplace_back(col_indA.size());

    taskData->inputs_count.emplace_back(matrixB.size());
    taskData->inputs_count.emplace_back(matrixB[0].size());
    taskData->inputs_count.emplace_back(dB.size());
    taskData->inputs_count.emplace_back(row_indB.size());
    taskData->inputs_count.emplace_back(col_indB.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < out.size(); ++i) {
      taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
    }
    taskData->outputs_count.emplace_back(out.size());
    taskData->outputs_count.emplace_back(out[0].size());
  }

  morozov_e_mult_sparse_matrix::TestMPITaskParallel testMpiTaskParallel(taskData);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::vector<double>> ans(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    for (size_t i = 0; i < out.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskData->outputs[i]);
      ans[i] = std::vector(ptr, ptr + matrixB[0].size());
    }
    std::vector<std::vector<double>> check_result = {{29, 0, 70}, {30, 12, 76}};
    ASSERT_EQ(check_result, ans);
  }
}
