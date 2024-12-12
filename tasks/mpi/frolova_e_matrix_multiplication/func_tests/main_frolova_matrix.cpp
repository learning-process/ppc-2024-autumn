// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/frolova_e_matrix_multiplication/include/ops_mpi_frolova_matrix.hpp"

void randomNumVec(int N, std::vector<int>& vec) {
  for (int i = 0; i < N; i++) {
    int num = rand() % 100 + 1;
    vec.push_back(num);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, multiplication_of_square_matrices) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {4, 4};
  std::vector<int> values_2 = {4, 4};
  std::vector<int> matrixA_;
  std::vector<int> matrixB_;
  std::vector<int32_t> res(16);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(16, matrixA_);
    randomNumVec(16, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_matrix(16);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, multiplication_of_two_vectors) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {1, 4};
  std::vector<int> values_2 = {4, 1};
  std::vector<int> matrixA_;
  std::vector<int> matrixB_;
  std::vector<int32_t> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(4, matrixA_);
    randomNumVec(4, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_matrix(1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, values_are_prime_numbers) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {61, 61};
  std::vector<int> values_2 = {61, 61};
  std::vector<int> matrixA_;
  std::vector<int> matrixB_;
  std::vector<int32_t> res(3721);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(3721, matrixA_);
    randomNumVec(3721, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_matrix(3721);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, matrix_with_single_element) {
  boost::mpi::communicator world;
  std::vector<int> values_1 = {1, 1};
  std::vector<int> values_2 = {1, 1};
  std::vector<int> matrixA_;
  std::vector<int> matrixB_;
  std::vector<int32_t> res(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(1, matrixA_);
    randomNumVec(1, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> reference_matrix(1);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, multiplication_of_large_matrices) {
  // creare data

  boost::mpi::communicator world;
  std::vector<int> values_1 = {20, 20};
  std::vector<int> values_2 = {20, 20};
  std::vector<int> matrixA_;

  std::vector<int> matrixB_;

  std::vector<int32_t> res(400);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(400, matrixA_);
    randomNumVec(400, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_matrix(400);

    // Create TaskData

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, matrix_of_size_of_power_of_two) {
  // creare data

  boost::mpi::communicator world;
  std::vector<int> values_1 = {64, 64};
  std::vector<int> values_2 = {64, 64};
  std::vector<int> matrixA_;

  std::vector<int> matrixB_;

  std::vector<int32_t> res(4096);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(4096, matrixA_);
    randomNumVec(4096, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_matrix(4096);

    // Create TaskData

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, matrices_multiplication_with_wound_of_string_length) {
  // creare data

  boost::mpi::communicator world;
  std::vector<int> values_1 = {2, 4};
  std::vector<int> values_2 = {4, 3};
  std::vector<int> matrixA_;

  std::vector<int> matrixB_;

  std::vector<int32_t> res(6);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(8, matrixA_);
    randomNumVec(12, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_matrix(6);

    // Create TaskData

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, multiplying_vector_by_a_matrix) {
  // creare data

  boost::mpi::communicator world;
  std::vector<int> values_1 = {1, 4};
  std::vector<int> values_2 = {4, 4};
  std::vector<int> matrixA_;

  std::vector<int> matrixB_;

  std::vector<int32_t> res(4);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(4, matrixA_);
    randomNumVec(16, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_matrix(4);

    // Create TaskData

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, multiplying_matrix_by_a_vector) {
  // creare data

  boost::mpi::communicator world;
  std::vector<int> values_1 = {4, 4};
  std::vector<int> values_2 = {4, 1};
  std::vector<int> matrixA_;

  std::vector<int> matrixB_;

  std::vector<int32_t> res(4);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(16, matrixA_);
    randomNumVec(4, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }

  frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<int32_t> reference_matrix(4);

    // Create TaskData

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataSeq->inputs_count.emplace_back(values_1.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataSeq->inputs_count.emplace_back(values_2.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixA_.size());

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataSeq->inputs_count.emplace_back(matrixB_.size());

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_matrix.data()));
    taskDataSeq->outputs_count.emplace_back(reference_matrix.size());

    // Create Task
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_matrix, res);
  }
}

//-----------ASSERT_FALSE

TEST(frolova_e_matrix_multiplication_mpi, value1_dont_have_two_elements) {
  // creare data

  boost::mpi::communicator world;
  std::vector<int> values_1 = {4};
  std::vector<int> values_2 = {4, 1};
  std::vector<int> matrixA_;

  std::vector<int> matrixB_;

  std::vector<int32_t> res(4);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(16, matrixA_);
    randomNumVec(4, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, value2_dont_have_two_elements) {
  // creare data

  boost::mpi::communicator world;
  std::vector<int> values_1 = {4, 4};
  std::vector<int> values_2 = {4};
  std::vector<int> matrixA_;

  std::vector<int> matrixB_;

  std::vector<int32_t> res(4);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(16, matrixA_);
    randomNumVec(4, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, rows_number_is_not_equal_to_the_columns_number) {
  // creare data

  boost::mpi::communicator world;
  std::vector<int> values_1 = {4, 1};
  std::vector<int> values_2 = {4, 1};
  std::vector<int> matrixA_;

  std::vector<int> matrixB_;

  std::vector<int32_t> res(4);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(16, matrixA_);
    randomNumVec(4, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(frolova_e_matrix_multiplication_mpi, mismatch_in_the_size_of_the_resulting_vector) {
  // creare data

  boost::mpi::communicator world;
  std::vector<int> values_1 = {4, 1};
  std::vector<int> values_2 = {4, 4};
  std::vector<int> matrixA_;

  std::vector<int> matrixB_;

  std::vector<int32_t> res(4);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    randomNumVec(16, matrixA_);
    randomNumVec(4, matrixB_);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_1.data()));
    taskDataPar->inputs_count.emplace_back(values_1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(values_2.data()));
    taskDataPar->inputs_count.emplace_back(values_2.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixA_.data()));
    taskDataPar->inputs_count.emplace_back(matrixA_.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrixB_.data()));
    taskDataPar->inputs_count.emplace_back(matrixB_.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
    frolova_e_matrix_multiplication_mpi::matrixMultiplicationParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}