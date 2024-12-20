// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdlib>
#include <vector>

#include "mpi/morozov_e_mult_sparse_matrix/include/ops_mpi.hpp"
namespace morozov_e_mult_sparse_matrix {
std::vector<std::vector<double>> generateRandomMatrix(int rows, int columns) {
  std::vector<std::vector<double>> result(rows, std::vector<double>(columns, 0));
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < columns; ++j) {
      double value = static_cast<double>(rand()) / RAND_MAX;
      result[i][j] = value * 100;
    }
  }
  return result;
}
}  // namespace morozov_e_mult_sparse_matrix
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
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> outPar(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  std::vector<std::vector<double>> outSeq(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    // parallel version
    taskDataPar->inputs_count.emplace_back(matrixA.size());
    taskDataPar->inputs_count.emplace_back(matrixA[0].size());
    taskDataPar->inputs_count.emplace_back(dA.size());
    taskDataPar->inputs_count.emplace_back(row_indA.size());
    taskDataPar->inputs_count.emplace_back(col_indA.size());
    taskDataPar->inputs_count.emplace_back(matrixB.size());
    taskDataPar->inputs_count.emplace_back(matrixB[0].size());
    taskDataPar->inputs_count.emplace_back(dB.size());
    taskDataPar->inputs_count.emplace_back(row_indB.size());
    taskDataPar->inputs_count.emplace_back(col_indB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < outPar.size(); ++i) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(outPar[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(outPar.size());
    taskDataPar->outputs_count.emplace_back(outPar[0].size());
    // seq version
    taskDataSeq->inputs_count.emplace_back(matrixA.size());
    taskDataSeq->inputs_count.emplace_back(matrixA[0].size());
    taskDataSeq->inputs_count.emplace_back(dA.size());
    taskDataSeq->inputs_count.emplace_back(row_indA.size());
    taskDataSeq->inputs_count.emplace_back(col_indA.size());
    taskDataSeq->inputs_count.emplace_back(matrixB.size());
    taskDataSeq->inputs_count.emplace_back(matrixB[0].size());
    taskDataSeq->inputs_count.emplace_back(dB.size());
    taskDataSeq->inputs_count.emplace_back(row_indB.size());
    taskDataSeq->inputs_count.emplace_back(col_indB.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < outPar.size(); ++i) {
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outSeq[i].data()));
    }
    taskDataSeq->outputs_count.emplace_back(outSeq.size());
    taskDataSeq->outputs_count.emplace_back(outSeq[0].size());
  }

  morozov_e_mult_sparse_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::vector<double>> ansPar(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    std::vector<std::vector<double>> ansSeq(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    for (size_t i = 0; i < outPar.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskDataPar->outputs[i]);
      ansPar[i] = std::vector(ptr, ptr + matrixB.size());
    }
    morozov_e_mult_sparse_matrix::TestTaskSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();
    for (size_t i = 0; i < outSeq.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskDataSeq->outputs[i]);
      ansSeq[i] = std::vector(ptr, ptr + matrixB.size());
    }
    ASSERT_EQ(ansSeq, ansPar);
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
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> outPar(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  std::vector<std::vector<double>> outSeq(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    // par version
    taskDataPar->inputs_count.emplace_back(matrixA.size());
    taskDataPar->inputs_count.emplace_back(matrixA[0].size());
    taskDataPar->inputs_count.emplace_back(dA.size());
    taskDataPar->inputs_count.emplace_back(row_indA.size());
    taskDataPar->inputs_count.emplace_back(col_indA.size());

    taskDataPar->inputs_count.emplace_back(matrixB.size());
    taskDataPar->inputs_count.emplace_back(matrixB[0].size());
    taskDataPar->inputs_count.emplace_back(dB.size());
    taskDataPar->inputs_count.emplace_back(row_indB.size());
    taskDataPar->inputs_count.emplace_back(col_indB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < outPar.size(); ++i) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(outPar[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(outPar.size());
    taskDataPar->outputs_count.emplace_back(outPar[0].size());
    // seq version
    taskDataSeq->inputs_count.emplace_back(matrixA.size());
    taskDataSeq->inputs_count.emplace_back(matrixA[0].size());
    taskDataSeq->inputs_count.emplace_back(dA.size());
    taskDataSeq->inputs_count.emplace_back(row_indA.size());
    taskDataSeq->inputs_count.emplace_back(col_indA.size());

    taskDataSeq->inputs_count.emplace_back(matrixB.size());
    taskDataSeq->inputs_count.emplace_back(matrixB[0].size());
    taskDataSeq->inputs_count.emplace_back(dB.size());
    taskDataSeq->inputs_count.emplace_back(row_indB.size());
    taskDataSeq->inputs_count.emplace_back(col_indB.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < outSeq.size(); ++i) {
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outSeq[i].data()));
    }
    taskDataSeq->outputs_count.emplace_back(outSeq.size());
    taskDataSeq->outputs_count.emplace_back(outSeq[0].size());
  }

  morozov_e_mult_sparse_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::vector<double>> ansPar(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    std::vector<std::vector<double>> ansSeq(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    for (size_t i = 0; i < outPar.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskDataPar->outputs[i]);
      ansPar[i] = std::vector(ptr, ptr + matrixB[0].size());
    }
    morozov_e_mult_sparse_matrix::TestTaskSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();
    for (size_t i = 0; i < outSeq.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskDataSeq->outputs[i]);
      ansSeq[i] = std::vector(ptr, ptr + matrixB[0].size());
    }
    ASSERT_EQ(ansSeq, ansPar);
  }
}
TEST(morozov_e_mult_sparse_matrix, Test_Main3) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrixA = {{0.2, 0, 0.5}, {0, 0.7, 6}, {0.1, 0, 0.8}};
  std::vector<std::vector<double>> matrixB = {{0.15, 0}, {0, 0.3}, {0.4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> outPar(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  std::vector<std::vector<double>> outSeq(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(matrixA.size());
    taskDataPar->inputs_count.emplace_back(matrixA[0].size());
    taskDataPar->inputs_count.emplace_back(dA.size());
    taskDataPar->inputs_count.emplace_back(row_indA.size());
    taskDataPar->inputs_count.emplace_back(col_indA.size());

    taskDataPar->inputs_count.emplace_back(matrixB.size());
    taskDataPar->inputs_count.emplace_back(matrixB[0].size());
    taskDataPar->inputs_count.emplace_back(dB.size());
    taskDataPar->inputs_count.emplace_back(row_indB.size());
    taskDataPar->inputs_count.emplace_back(col_indB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < outPar.size(); ++i) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(outPar[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(outPar.size());
    taskDataPar->outputs_count.emplace_back(outPar[0].size());
    // seq version
    taskDataSeq->inputs_count.emplace_back(matrixA.size());
    taskDataSeq->inputs_count.emplace_back(matrixA[0].size());
    taskDataSeq->inputs_count.emplace_back(dA.size());
    taskDataSeq->inputs_count.emplace_back(row_indA.size());
    taskDataSeq->inputs_count.emplace_back(col_indA.size());

    taskDataSeq->inputs_count.emplace_back(matrixB.size());
    taskDataSeq->inputs_count.emplace_back(matrixB[0].size());
    taskDataSeq->inputs_count.emplace_back(dB.size());
    taskDataSeq->inputs_count.emplace_back(row_indB.size());
    taskDataSeq->inputs_count.emplace_back(col_indB.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < outSeq.size(); ++i) {
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outSeq[i].data()));
    }
    taskDataSeq->outputs_count.emplace_back(outSeq.size());
    taskDataSeq->outputs_count.emplace_back(outSeq[0].size());
  }

  morozov_e_mult_sparse_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::vector<double>> ansPar(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    std::vector<std::vector<double>> ansSeq(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    for (size_t i = 0; i < outPar.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskDataPar->outputs[i]);
      ansPar[i] = std::vector(ptr, ptr + matrixB[0].size());
    }
    morozov_e_mult_sparse_matrix::TestTaskSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();
    for (size_t i = 0; i < outSeq.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskDataSeq->outputs[i]);
      ansSeq[i] = std::vector(ptr, ptr + matrixB[0].size());
    }
    ASSERT_EQ(ansSeq, ansPar);
  }
}
TEST(morozov_e_mult_sparse_matrix, Test_Main4) {
  boost::mpi::communicator world;
  std::vector<std::vector<double>> matrixA = morozov_e_mult_sparse_matrix::generateRandomMatrix(2, 3);
  std::vector<std::vector<double>> matrixB = morozov_e_mult_sparse_matrix::generateRandomMatrix(3, 2);
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<double>> outPar(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  std::vector<std::vector<double>> outSeq(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(matrixA.size());
    taskDataPar->inputs_count.emplace_back(matrixA[0].size());
    taskDataPar->inputs_count.emplace_back(dA.size());
    taskDataPar->inputs_count.emplace_back(row_indA.size());
    taskDataPar->inputs_count.emplace_back(col_indA.size());

    taskDataPar->inputs_count.emplace_back(matrixB.size());
    taskDataPar->inputs_count.emplace_back(matrixB[0].size());
    taskDataPar->inputs_count.emplace_back(dB.size());
    taskDataPar->inputs_count.emplace_back(row_indB.size());
    taskDataPar->inputs_count.emplace_back(col_indB.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < outPar.size(); ++i) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(outPar[i].data()));
    }
    taskDataPar->outputs_count.emplace_back(outPar.size());
    taskDataPar->outputs_count.emplace_back(outPar[0].size());
    // seq version
    taskDataSeq->inputs_count.emplace_back(matrixA.size());
    taskDataSeq->inputs_count.emplace_back(matrixA[0].size());
    taskDataSeq->inputs_count.emplace_back(dA.size());
    taskDataSeq->inputs_count.emplace_back(row_indA.size());
    taskDataSeq->inputs_count.emplace_back(col_indA.size());

    taskDataSeq->inputs_count.emplace_back(matrixB.size());
    taskDataSeq->inputs_count.emplace_back(matrixB[0].size());
    taskDataSeq->inputs_count.emplace_back(dB.size());
    taskDataSeq->inputs_count.emplace_back(row_indB.size());
    taskDataSeq->inputs_count.emplace_back(col_indB.size());
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));

    for (size_t i = 0; i < outSeq.size(); ++i) {
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(outSeq[i].data()));
    }
    taskDataSeq->outputs_count.emplace_back(outSeq.size());
    taskDataSeq->outputs_count.emplace_back(outSeq[0].size());
  }

  morozov_e_mult_sparse_matrix::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<std::vector<double>> ansPar(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    std::vector<std::vector<double>> ansSeq(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
    for (size_t i = 0; i < outPar.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskDataPar->outputs[i]);
      ansPar[i] = std::vector(ptr, ptr + matrixB[0].size());
    }
    morozov_e_mult_sparse_matrix::TestTaskSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq.validation(), true);
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();
    for (size_t i = 0; i < outSeq.size(); ++i) {
      auto *ptr = reinterpret_cast<double *>(taskDataSeq->outputs[i]);
      ansSeq[i] = std::vector(ptr, ptr + matrixB[0].size());
    }
    ASSERT_EQ(ansSeq, ansPar);
  }
}