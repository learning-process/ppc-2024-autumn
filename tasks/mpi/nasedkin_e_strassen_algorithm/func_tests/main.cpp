#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iomanip>
#include <random>
#include <vector>

#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

std::vector<double> generateRandomMatrix(int size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-100.0, 100.0);
  std::vector<double> matrix(size * size);

  if (size <= 0) {
    std::cout << "generateRandomMatrix: Invalid size: " << size << std::endl;
    return matrix;
  }

  for (int i = 0; i < size * size; i++) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_2x2) {
  boost::mpi::communicator world;

  int matrixSize = 2;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA.size());
    taskDataSeq->inputs_count.emplace_back(matrixB.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resultSeq.size());

    std::cout << "Test: TaskData inputs_count[0] = " << taskDataSeq->inputs_count[0]
              << ", inputs_count[1] = " << taskDataSeq->inputs_count[1] << std::endl;

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
    std::cout << "SEQ Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());

    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }
  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  ASSERT_EQ(resultSeq, resultParallel);
  std::cout << "Parallel Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_4x4) {
  boost::mpi::communicator world;

  int matrixSize = 4;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA.size());
    taskDataSeq->inputs_count.emplace_back(matrixB.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resultSeq.size());

    std::cout << "Test: TaskData inputs_count[0] = " << taskDataSeq->inputs_count[0]
              << ", inputs_count[1] = " << taskDataSeq->inputs_count[1] << std::endl;

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
    std::cout << "Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());

    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }
  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  std::cout << "Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;

  ASSERT_EQ(resultSeq, resultParallel);
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_8x8) {
  boost::mpi::communicator world;

  int matrixSize = 8;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA.size());
    taskDataSeq->inputs_count.emplace_back(matrixB.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resultSeq.size());

    std::cout << "Test: TaskData inputs_count[0] = " << taskDataSeq->inputs_count[0]
              << ", inputs_count[1] = " << taskDataSeq->inputs_count[1] << std::endl;

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
    std::cout << "Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());

    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }
  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  std::cout << "Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;

  ASSERT_EQ(resultSeq, resultParallel);
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_16x16) {
  boost::mpi::communicator world;

  int matrixSize = 16;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA.size());
    taskDataSeq->inputs_count.emplace_back(matrixB.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resultSeq.size());

    std::cout << "Test: TaskData inputs_count[0] = " << taskDataSeq->inputs_count[0]
              << ", inputs_count[1] = " << taskDataSeq->inputs_count[1] << std::endl;

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
    std::cout << "Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());

    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }
  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  std::cout << "Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;

  ASSERT_EQ(resultSeq, resultParallel);
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_32x32) {
  boost::mpi::communicator world;

  int matrixSize = 32;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);
  std::vector<double> resultSeq(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA.size());
    taskDataSeq->inputs_count.emplace_back(matrixB.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultSeq.data()));
    taskDataSeq->outputs_count.emplace_back(resultSeq.size());

    std::cout << "Test: TaskData inputs_count[0] = " << taskDataSeq->inputs_count[0]
              << ", inputs_count[1] = " << taskDataSeq->inputs_count[1] << std::endl;

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
    std::cout << "Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());

    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }
  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  std::cout << "Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std::endl;

  ASSERT_EQ(resultSeq, resultParallel);
}