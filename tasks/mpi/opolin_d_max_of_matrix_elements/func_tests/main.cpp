// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/opolin_d_max_of_matrix_elements/include/ops_mpi.hpp"

std::vector<int> getRandomVectorForGetMaxInMatrix(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = min + gen() % (max - min + 1);
  }
  return vec;
}

std::vector<std::vector<int>> getRandomMatrixForGetMaxInMatrix(int rows, int cols, int min, int max) {
  std::vector<std::vector<int>> matr(rows);
  for (int i = 0; i < rows; i++) {
    matr[i] = getRandomVectorForGetMaxInMatrix(cols, min, max);
  }
  return matr;
}

TEST(opolin_d_max_of_matrix_elements_mpi, Test_Max_Matrix_1000x1000) {
  const int rows = 1000;
  const int cols = 1000;
  const int max = 1000;
  const int min = 0;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> matrix;
  std::vector<int32_t> out(1, std::numeric_limits<int32_t>::min());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix = getRandomMatrixForGetMaxInMatrix(rows, cols, min, max);
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> reference_max(1, std::numeric_limits<int32_t>::min());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max[0], out[0]);
  }
}

TEST(opolin_d_max_of_matrix_elements_mpi, Test_Max_Matrix_1x100) {
  const int rows = 1;
  const int cols = 100;
  const int max = 1000;
  const int min = 0;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> matrix;
  std::vector<int32_t> out(1, std::numeric_limits<int32_t>::min());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix = getRandomMatrixForGetMaxInMatrix(rows, cols, min, max);
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_max(1, std::numeric_limits<int32_t>::min());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], out[0]);
  }
}

TEST(opolin_d_max_of_matrix_elements_mpi, Test_Max_Matrix_100x1) {
  const int rows = 100;
  const int cols = 1;
  const int max = 1000;
  const int min = 0;
  boost::mpi::communicator world;
  std::vector<std::vector<int>> matrix;
  std::vector<int32_t> out(1, std::numeric_limits<int32_t>::min());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix = getRandomMatrixForGetMaxInMatrix(rows, cols, min, max);
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    std::vector<int32_t> reference_max(1, std::numeric_limits<int32_t>::min());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], out[0]);
  }
}

TEST(opolin_d_max_of_matrix_elements_mpi, Test_Max_Matrix_Negative) {
  const int rows = 10;
  const int cols = 10;
  const int min = -100;
  const int max = -1;

  boost::mpi::communicator world;
  std::vector<std::vector<int>> matrix;
  std::vector<int32_t> out(1, std::numeric_limits<int32_t>::min());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    matrix = getRandomMatrixForGetMaxInMatrix(rows, cols, min, max);
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataPar->inputs_count.emplace_back(rows);
    taskDataPar->inputs_count.emplace_back(cols);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  opolin_d_max_of_matrix_elements_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_max(1, std::numeric_limits<int32_t>::min());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    for (unsigned int i = 0; i < matrix.size(); i++)
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
    taskDataSeq->inputs_count.emplace_back(rows);
    taskDataSeq->inputs_count.emplace_back(cols);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());
    opolin_d_max_of_matrix_elements_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference_max[0], out[0]);
  }
}