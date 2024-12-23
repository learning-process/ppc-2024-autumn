// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_mpi.hpp"

static std::vector<double> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);
  std::vector<double> vec(sz);
  for (int i = 0; i < sz; ++i) {
    vec[i] = dist(gen);
  }
  return vec;
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, Test_Multiplication) {
  boost::mpi::communicator world;
  int matrix_size = 2;
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> B = {5.0, 6.0, 7.0, 8.0};
  std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);
  std::vector<double> C_sequential(matrix_size * matrix_size, 0.0);
  size_t block_size = 1;
  std::vector<double> C_expected = {19.0, 22.0, 43.0, 50.0};
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
  }
  lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
    taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs_count.emplace_back(matrix_size * matrix_size);
    lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskSequential matrixMultiplication(taskDataSeq);
    ASSERT_EQ(matrixMultiplication.validation(), true);
    matrixMultiplication.pre_processing();
    matrixMultiplication.run();
    matrixMultiplication.post_processing();
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
      ASSERT_NEAR(C_sequential[i], C_parallel[i], 1e-9);
    }
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, Test_Multiplication_0x0) {
  boost::mpi::communicator world;
  int matrix_size = 0;
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> C_parallel;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
  }

  lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, RandomTestMatrix3x3) {
  boost::mpi::communicator world;
  int matrix_size = 20;
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);
  std::vector<double> C_sequential(matrix_size * matrix_size, 0.0);
  size_t block_size = 1;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    A = getRandomVector(matrix_size * matrix_size);
    B = getRandomVector(matrix_size * matrix_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
  }
  lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
    taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs_count.emplace_back(matrix_size * matrix_size);
    lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskSequential matrixMultiplication(taskDataSeq);
    ASSERT_EQ(matrixMultiplication.validation(), true);
    matrixMultiplication.pre_processing();
    matrixMultiplication.run();
    matrixMultiplication.post_processing();
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
      ASSERT_NEAR(C_sequential[i], C_parallel[i], 1e-9);
    }
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, RandomTestMatrix13x13) {
  boost::mpi::communicator world;
  int matrix_size = 20;
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);
  std::vector<double> C_sequential(matrix_size * matrix_size, 0.0);
  size_t block_size = 1;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    A = getRandomVector(matrix_size * matrix_size);
    B = getRandomVector(matrix_size * matrix_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
  }
  lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
    taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs_count.emplace_back(matrix_size * matrix_size);
    lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskSequential matrixMultiplication(taskDataSeq);
    ASSERT_EQ(matrixMultiplication.validation(), true);
    matrixMultiplication.pre_processing();
    matrixMultiplication.run();
    matrixMultiplication.post_processing();
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
      ASSERT_NEAR(C_sequential[i], C_parallel[i], 1e-9);
    }
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, RandomTest) {
  boost::mpi::communicator world;
  int matrix_size = 20;
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);
  std::vector<double> C_sequential(matrix_size * matrix_size, 0.0);
  size_t block_size = 1;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    A = getRandomVector(matrix_size * matrix_size);
    B = getRandomVector(matrix_size * matrix_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
  }
  lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C_sequential.data()));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
    taskDataSeq->inputs_count.emplace_back(matrix_size * matrix_size);
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs_count.emplace_back(matrix_size * matrix_size);
    lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskSequential matrixMultiplication(taskDataSeq);
    ASSERT_EQ(matrixMultiplication.validation(), true);
    matrixMultiplication.pre_processing();
    matrixMultiplication.run();
    matrixMultiplication.post_processing();
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
      ASSERT_NEAR(C_sequential[i], C_parallel[i], 1e-9);
    }
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, Incorrect_input_data) {
  boost::mpi::communicator world;
  int matrix_size = 20;
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
  }
  lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, Incorrect_input_data2) {
  boost::mpi::communicator world;
  int matrix_size = -20;
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->inputs_count.emplace_back(sizeof(A.size()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
  }
  lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, Incorrect_output_data1) {
  boost::mpi::communicator world;
  int matrix_size = 20;
  std::vector<double> A(matrix_size * matrix_size, 0.0);
  std::vector<double> B(matrix_size * matrix_size, 0.0);
  std::vector<double> C_parallel(matrix_size * matrix_size, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->outputs_count.emplace_back(C_parallel.size() + 1);
  }
  lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_mpi, Incorrect_output_data2) {
  boost::mpi::communicator world;
  int matrix_size = 20;
  std::vector<double> A(matrix_size * matrix_size, 0.0);
  std::vector<double> B(matrix_size * matrix_size, 0.0);
  std::vector<double> C_parallel;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&matrix_size));
    taskDataPar->inputs_count.emplace_back(A.size());
    taskDataPar->inputs_count.emplace_back(B.size());
    taskDataPar->inputs_count.emplace_back(sizeof(matrix_size));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&C_parallel));
    taskDataPar->outputs_count.emplace_back(C_parallel.size());
  }
  lysov_i_matrix_multiplication_Fox_algorithm_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
  }
}