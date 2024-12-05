﻿
#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/Odintsov_M_VerticalRibbon_mpi/include/ops_mpi.hpp"

std::vector<double> getMatrix(int sz) {
  std::vector<double> matrix(sz, 0);
  int min = -100;
  int max = 100;
  srand(time(NULL));
  for (int i = 0; i < sz; ++i) {
    matrix[i] = min + rand() % (max - min + 1);
  }
  return matrix;
}

TEST(Odintsov_M_VerticalRibbon_mpi, sz_4) {
  // Create data
  boost::mpi::communicator com;

  // Create data
  std::vector<double> matrixA = {1, 2, 3, 4};
  std::vector<double> vectorB = {1, 2};
  std::vector<double> out(2, 0);
  std::vector<double> out_s(2, 0);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
    taskDataPar->inputs_count.emplace_back(4);
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->inputs_count.emplace_back(2);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  // Create Task
  Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPIParallel testClassPar(taskDataPar);
  ASSERT_TRUE(testClassPar.validation());
  testClassPar.pre_processing();
  testClassPar.run();
  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
    taskDataSeq->inputs_count.emplace_back(4);
    taskDataSeq->inputs_count.emplace_back(2);
    taskDataSeq->inputs_count.emplace_back(2);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(2);
    Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPISequential testClassSeq(taskDataSeq);
    ASSERT_TRUE(testClassSeq.validation());
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    EXPECT_EQ(out, out_s);
  }
}

TEST(Odintsov_M_VerticalRibbon_mpi, sz_3600) {
  // Create data
  boost::mpi::communicator com;

  // Create data
  std::vector<double> matrixA = getMatrix(3600);
  std::vector<double> vectorB = getMatrix(60);
  std::vector<double> out(60, 0);
  std::vector<double> out_s(60, 0);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
    taskDataPar->inputs_count.emplace_back(3600);
    taskDataPar->inputs_count.emplace_back(60);
    taskDataPar->inputs_count.emplace_back(60);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(60);
  }

  // Create Task
  Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPIParallel testClassPar(taskDataPar);
  ASSERT_TRUE(testClassPar.validation());
  testClassPar.pre_processing();

  testClassPar.run();

  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
    taskDataSeq->inputs_count.emplace_back(3600);
    taskDataSeq->inputs_count.emplace_back(60);
    taskDataSeq->inputs_count.emplace_back(60);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(60);
    Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPISequential testClassSeq(taskDataSeq);
    ASSERT_TRUE(testClassSeq.validation());
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    EXPECT_EQ(out, out_s);
  }
}

TEST(Odintsov_M_VerticalRibbon_mpi, sz_1800) {
  // Create data
  boost::mpi::communicator com;

  // Create data
  std::vector<double> matrixA = getMatrix(1800);
  std::vector<double> matrixB = getMatrix(60);
  std::vector<double> out(60, 0);
  std::vector<double> out_s(60, 0);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataPar->inputs_count.emplace_back(1800);
    taskDataPar->inputs_count.emplace_back(30);
    taskDataPar->inputs_count.emplace_back(60);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(60);
  }

  // Create Task
  Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPIParallel testClassPar(taskDataPar);
  ASSERT_TRUE(testClassPar.validation());
  testClassPar.pre_processing();

  testClassPar.run();

  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataSeq->inputs_count.emplace_back(1800);
    taskDataSeq->inputs_count.emplace_back(30);
    taskDataSeq->inputs_count.emplace_back(60);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(60);
    Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPISequential testClassSeq(taskDataSeq);
    ASSERT_TRUE(testClassSeq.validation());
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    EXPECT_EQ(out, out_s);
  }
}

TEST(Parallel_MPI_matrix, double_val) {
  // Create data
  boost::mpi::communicator com;

  // Create data
  std::vector<double> matrixA(144, 1.1);
  std::vector<double> vectorB(12, 1.1);
  std::vector<double> out(12, 0);
  std::vector<double> out_s(12, 0);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
    taskDataPar->inputs_count.emplace_back(144);
    taskDataPar->inputs_count.emplace_back(12);
    taskDataPar->inputs_count.emplace_back(12);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(12);
  }

  // Create Task
  Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPIParallel testClassPar(taskDataPar);
  ASSERT_TRUE(testClassPar.validation());
  testClassPar.pre_processing();

  testClassPar.run();

  testClassPar.post_processing();

  if (com.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vectorB.data()));
    taskDataSeq->inputs_count.emplace_back(144);
    taskDataSeq->inputs_count.emplace_back(12);
    taskDataSeq->inputs_count.emplace_back(12);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_s.data()));
    taskDataSeq->outputs_count.emplace_back(12);
    Odintsov_M_VerticalRibbon_mpi::VerticalRibbonMPISequential testClassSeq(taskDataSeq);
    ASSERT_TRUE(testClassSeq.validation());
    testClassSeq.pre_processing();
    testClassSeq.run();
    testClassSeq.post_processing();
    for (size_t i = 0; i < out.size(); i++) {
      EXPECT_NEAR(out[i], out_s[i], 1e-6);
    }
  }
}