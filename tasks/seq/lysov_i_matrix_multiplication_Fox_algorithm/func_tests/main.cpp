// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/lysov_i_matrix_multiplication_Fox_algorithm/include/ops_seq.hpp"

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_Matrix_Multiplication_Identity) {
  size_t N = 3;
  size_t block_size = 1;
  std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> B = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  std::vector<double> C(N * N, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.validation(), true);
  matrixMultiplication.pre_processing();
  matrixMultiplication.run();
  matrixMultiplication.post_processing();
  EXPECT_EQ(C, A);
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_Matrix_Multiplication_Arbitrary_Values) {
  size_t N = 3;
  size_t block_size = 1;
  std::vector<double> A = {2, 3, 1, 4, 0, 5, 1, 2, 3};
  std::vector<double> B = {1, 2, 3, 0, 1, 0, 4, 0, 1};
  std::vector<double> C(N * N, 0);
  std::vector<double> expected_C = {6.0, 7.0, 7.0, 24.0, 8.0, 17.0, 13.0, 4.0, 6.0};
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.validation(), true);
  matrixMultiplication.pre_processing();
  matrixMultiplication.run();
  matrixMultiplication.post_processing();
  EXPECT_EQ(C, expected_C);
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_Matrix_Multiplication_Invalid_Inputs) {
  size_t N = 3;
  size_t block_size = 1;
  std::vector<double> A = {2, 3, 1, 4, 0, 5};
  std::vector<double> B = {1, 2, 3, 0, 1, 0, 4, 0, 1};
  std::vector<double> C(N * N, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->inputs_count.emplace_back(2 * N);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.validation(), false);
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_Matrix_Multiplication_Empty_Matrix) {
  size_t N = 3;
  size_t block_size = 1;
  std::vector<double> A;
  std::vector<double> B = {1, 2, 3, 0, 1, 0, 4, 0, 1};
  std::vector<double> C(N * N, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->inputs_count.emplace_back(N * N);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs_count.emplace_back(N * N);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.validation(), false);
}

TEST(lysov_i_matrix_multiplication_Fox_algorithm_seq, Test_Matrix_Multiplication_IncompleteInput) {
  size_t N = 3;
  size_t block_size = 1;
  std::vector<double> A = {1, 2, 3, 4, 5, 6};
  std::vector<double> B = {9, 8, 7, 6, 5, 4};
  std::vector<double> C(N * N, 0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&N));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&block_size));
  taskDataSeq->inputs_count.emplace_back(3);
  taskDataSeq->inputs_count.emplace_back(6);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs_count.emplace_back(1);
  lysov_i_matrix_multiplication_Fox_algorithm_seq::TestTaskSequential matrixMultiplication(taskDataSeq);
  ASSERT_EQ(matrixMultiplication.validation(), false);
}
