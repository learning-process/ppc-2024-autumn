// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/frolova_e_matrix_multiplication/include/ops_seq_frolova_matrix.hpp"

TEST(frolova_e_matrix_multiplication_seq, singleElementMatrices) {
  std::vector<int> values_1 = {1, 1};
  std::vector<int> values_2 = {1, 1};
  std::vector<int> matrixA_ = {1};
  std::vector<int> matrixB_ = {2};
  std::vector<int> res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  frolova_e_matrix_multiplication_seq::matrixMultiplication testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(2, res[0]);
}

TEST(frolova_e_matrix_multiplication_seq, theSquareMatricesOfTheSameSize_1) {
  std::vector<int> values_1 = {2, 2};
  std::vector<int> values_2 = {2, 2};
  std::vector<int> matrixA_ = {1, 1, 1, 1};
  std::vector<int> matrixB_ = {1, 1, 1, 1};
  std::vector<int> resMatrix = {2, 2, 2, 2};
  std::vector<int> out(4);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  frolova_e_matrix_multiplication_seq::matrixMultiplication testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(resMatrix, out);
}

TEST(frolova_e_matrix_multiplication_seq, theSquareMatricesOfTheSameSize_2) {
  std::vector<int> values_1 = {3, 3};
  std::vector<int> values_2 = {3, 3};
  std::vector<int> matrixA_ = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> matrixB_ = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> resMatrix = {3, 3, 3, 3, 3, 3, 3, 3, 3};
  std::vector<int> out(9);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  frolova_e_matrix_multiplication_seq::matrixMultiplication testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(resMatrix, out);
}

TEST(frolova_e_matrix_multiplication_seq, rectangularMatriceAndtheSquareMatrice) {
  std::vector<int> values_1 = {2, 3};
  std::vector<int> values_2 = {3, 3};
  std::vector<int> matrixA_ = {1, 1, 1, 1, 1, 1};
  std::vector<int> matrixB_ = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> resMatrix = {3, 3, 3, 3, 3, 3};
  std::vector<int> out(6);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  frolova_e_matrix_multiplication_seq::matrixMultiplication testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(resMatrix, out);
}

TEST(frolova_e_matrix_multiplication_seq, twoRectangularMultiplicationMatrices) {
  std::vector<int> values_1 = {2, 3};
  std::vector<int> values_2 = {3, 4};
  std::vector<int> matrixA_ = {1, 1, 1, 1, 1, 1};
  std::vector<int> matrixB_ = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  std::vector<int> resMatrix = {3, 3, 3, 3, 3, 3, 3, 3};
  std::vector<int> out(8);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  frolova_e_matrix_multiplication_seq::matrixMultiplication testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(resMatrix, out);
}

TEST(frolova_e_matrix_multiplication_seq, multiplicationOfTwoVectors) {
  std::vector<int> values_1 = {1, 3};
  std::vector<int> values_2 = {3, 1};
  std::vector<int> matrixA_ = {1, 1, 1};
  std::vector<int> matrixB_ = {1, 1, 1};
  std::vector<int> resMatrix = {3};
  std::vector<int> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  frolova_e_matrix_multiplication_seq::matrixMultiplication testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(resMatrix, out);
}

TEST(frolova_e_matrix_multiplication_seq, theNumberOfColumnsDoesNotMatchTheNumberOfRows) {
  std::vector<int> values_1 = {2, 3};
  std::vector<int> values_2 = {2, 1};
  std::vector<int> matrixA_ = {1, 1, 1, 1, 1, 1};
  std::vector<int> matrixB_ = {1, 1};
  std::vector<int> resMatrix = {3};
  std::vector<int> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  frolova_e_matrix_multiplication_seq::matrixMultiplication testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(frolova_e_matrix_multiplication_seq, transmittingIncorrectValues_1) {
  std::vector<int> values_1 = {0, 3};
  std::vector<int> values_2 = {2, 1};
  std::vector<int> matrixA_ = {1, 1, 1, 1, 1, 1};
  std::vector<int> matrixB_ = {1, 1};
  std::vector<int> resMatrix = {3};
  std::vector<int> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  frolova_e_matrix_multiplication_seq::matrixMultiplication testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(frolova_e_matrix_multiplication_seq, transmittingIncorrectValues_2) {
  std::vector<int> values_1 = {1, 3};
  std::vector<int> values_2 = {2, 1};
  std::vector<int> matrixA_ = {1, 1, 1, 1, 1, 1};
  std::vector<int> matrixB_ = {1, 1};
  std::vector<int> resMatrix = {3};
  std::vector<int> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_1.data()));
  taskDataSeq->inputs_count.emplace_back(values_1.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values_2.data()));
  taskDataSeq->inputs_count.emplace_back(values_2.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixA_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixA_.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrixB_.data()));
  taskDataSeq->inputs_count.emplace_back(matrixB_.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  frolova_e_matrix_multiplication_seq::matrixMultiplication testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}