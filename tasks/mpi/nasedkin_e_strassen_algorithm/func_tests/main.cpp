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
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  if (world.rank() == 0) {
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();


  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  if (world.rank() == 0) {
    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }

  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());
  std::cout<< "Test for " << matrixSize << "x" << matrixSize << " matrix finished" << std:endl;
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_3x3) {
  boost::mpi::communicator world;

  int matrixSize = 3;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  if (world.rank() == 0) {
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_4x4) {
  boost::mpi::communicator world;

  int matrixSize = 4;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  if (world.rank() == 0) {
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_5x5) {
  boost::mpi::communicator world;

  int matrixSize = 5;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  if (world.rank() == 0) {
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_8x8) {
  boost::mpi::communicator world;

  int matrixSize = 8;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  if (world.rank() == 0) {
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_16x16) {
  boost::mpi::communicator world;

  int matrixSize = 16;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  if (world.rank() == 0) {
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_32x32) {
  boost::mpi::communicator world;

  int matrixSize = 32;
  std::vector<double> matrixA = generateRandomMatrix(matrixSize);
  std::vector<double> matrixB = generateRandomMatrix(matrixSize);
  if (world.rank() == 0) {
    std::cout << "Test: MatrixA size = " << matrixSize << ", MatrixB size = " << matrixSize << std::endl;
  }
  std::vector<double> resultParallel(matrixSize * matrixSize, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::cout << "Test: TaskData inputs_count[0] = " << taskDataParallel->inputs_count[0]
              << ", inputs_count[1] = " << taskDataParallel->inputs_count[1] << std::endl;
  }

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskDataParallel->inputs_count.emplace_back(matrixA.size());
    taskDataParallel->inputs_count.emplace_back(matrixB.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(resultParallel.data()));
    taskDataParallel->outputs_count.emplace_back(resultParallel.size());
  }

  nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI testMpiTaskParallel(taskDataParallel);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  ASSERT_TRUE(testMpiTaskParallel.pre_processing());
  ASSERT_TRUE(testMpiTaskParallel.run());
  ASSERT_TRUE(testMpiTaskParallel.post_processing());

}