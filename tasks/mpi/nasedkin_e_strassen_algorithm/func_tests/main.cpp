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

  for (int i = 0; i < size * size; i++) {
    matrix[i] = dis(gen);
  }
  return matrix;
}

TEST(nasedkin_e_strassen_algorithm_mpi, EmptyInputs) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs = {};

    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI task(taskData);
    ASSERT_FALSE(task.validation());
  }
}

TEST(nasedkin_e_strassen_algorithm_mpi, MismatchedMatrixSizes) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto taskData = std::make_shared<ppc::core::TaskData>();

    std::vector<double> matrixA = {1, 2, 3, 4};
    std::vector<double> matrixB = {1, 2, 3, 4, 5, 6, 7, 8, 9};

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskData->inputs_count.emplace_back(matrixA.size());
    taskData->inputs_count.emplace_back(matrixB.size());
    taskData->outputs.emplace_back(nullptr);
    taskData->outputs_count.emplace_back(0);

    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI task(taskData);
    ASSERT_FALSE(task.validation());
  }
}

TEST(nasedkin_e_strassen_algorithm_mpi, InvalidOutputSize) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    auto taskData = std::make_shared<ppc::core::TaskData>();

    std::vector<double> matrixA = {1, 2, 3, 4};
    std::vector<double> matrixB = {5, 6, 7, 8};
    std::vector<double> result(6);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB.data()));
    taskData->inputs_count.emplace_back(matrixA.size());
    taskData->inputs_count.emplace_back(matrixB.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskData->outputs_count.emplace_back(result.size());

    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI task(taskData);
    ASSERT_FALSE(task.validation());
  }
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_2x2) {
  boost::mpi::communicator world;

  int matrixSize = 2;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
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

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

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
  ASSERT_EQ(resultSeq, resultParallel);
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_3x3) {
  boost::mpi::communicator world;

  int matrixSize = 3;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
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

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

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
  ASSERT_EQ(resultSeq, resultParallel);
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_4x4) {
  boost::mpi::communicator world;

  int matrixSize = 4;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
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

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

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

  ASSERT_EQ(resultSeq, resultParallel);
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_7x7) {
  boost::mpi::communicator world;

  int matrixSize = 7;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
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

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

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

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

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

  ASSERT_EQ(resultSeq, resultParallel);
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_10x10) {
  boost::mpi::communicator world;

  int matrixSize = 10;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
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

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

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

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

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

  ASSERT_EQ(resultSeq, resultParallel);
}

TEST(nasedkin_e_strassen_algorithm_mpi, Test_17x17) {
  boost::mpi::communicator world;

  int matrixSize = 17;
  std::vector<double> matrixA;
  std::vector<double> matrixB;
  if (world.rank() == 0) {
    matrixA = generateRandomMatrix(matrixSize);
    matrixB = generateRandomMatrix(matrixSize);
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

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

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

    nasedkin_e_strassen_algorithm::StrassenAlgorithmSEQ testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    ASSERT_TRUE(testMpiTaskSeq.pre_processing());
    ASSERT_TRUE(testMpiTaskSeq.run());
    ASSERT_TRUE(testMpiTaskSeq.post_processing());
  }

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

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

  ASSERT_EQ(resultSeq, resultParallel);
}
