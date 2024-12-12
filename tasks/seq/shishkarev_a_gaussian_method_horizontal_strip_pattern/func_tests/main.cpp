#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "seq/shishkarev_a_gaussian_method_horizontal_strip_pattern/include/ops_seq.hpp"

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_for_empty_matrix) {
  const int cols = 0;
  const int rows = 0;

  std::vector<double> matrix;
  std::vector<double> res;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_FALSE(MPIGaussHorizontalSequential.validation());
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_for_matrix_with_one_element) {
  const int cols = 1;
  const int rows = 1;

  std::vector<double> matrix = {1};
  std::vector<double> res;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_FALSE(MPIGaussHorizontalSequential.validation());
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_not_square_matrix) {
  const int cols = 5;
  const int rows = 2;

  std::vector<double> matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<double> res(cols - 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_FALSE(MPIGaussHorizontalSequential.validation());
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_zero_determinant) {
  const int cols = 4;
  const int rows = 3;

  std::vector<double> matrix = {6, -1, 12, 3, -3, -5, -6, 9, 1, 4, 2, -1};
  std::vector<double> res(cols - 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_FALSE(MPIGaussHorizontalSequential.validation());
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_not_equal_rank) {
  const int cols = 4;
  const int rows = 3;

  std::vector<double> matrix = {1, 2, 3, 7, 4, 5, 6, 2, 7, 8, 9, 8};
  std::vector<double> res(cols - 1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_FALSE(MPIGaussHorizontalSequential.validation());
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_3x2) {
  const int cols = 3;
  const int rows = 2;

  std::vector<double> matrix = {1, -1, -5, 2, 1, -7};
  std::vector<double> res(cols - 1, 0);
  std::vector<double> reference_res = {-4, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_EQ(MPIGaussHorizontalSequential.validation(), true);
  MPIGaussHorizontalSequential.pre_processing();
  MPIGaussHorizontalSequential.run();
  MPIGaussHorizontalSequential.post_processing();
  ASSERT_EQ(reference_res, res);
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_4x3) {
  const int cols = 4;
  const int rows = 3;

  std::vector<double> matrix = {3, 2, -5, -1, 2, -1, 3, 13, 1, 2, -1, 9};
  std::vector<double> res(cols - 1, 0);
  std::vector<double> reference_res = {3, 5, 4};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_EQ(MPIGaussHorizontalSequential.validation(), true);
  MPIGaussHorizontalSequential.pre_processing();
  MPIGaussHorizontalSequential.run();
  MPIGaussHorizontalSequential.post_processing();
  ASSERT_EQ(reference_res, res);
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_5x4) {
  const int cols = 5;
  const int rows = 4;

  std::vector<double> matrix = {1, 1, 2, 3, 1, 1, 2, 3, -1, -4, 3, -1, -1, -2, -4, 2, 3, -1, -1, -6};
  std::vector<double> res(cols - 1, 0);
  std::vector<double> reference_res = {-1, -1, 0, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_EQ(MPIGaussHorizontalSequential.validation(), true);
  MPIGaussHorizontalSequential.pre_processing();
  MPIGaussHorizontalSequential.run();
  MPIGaussHorizontalSequential.post_processing();
  ASSERT_EQ(reference_res, res);
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_11x10) {
  const int cols = 11;
  const int rows = 10;

  std::vector<double> matrix(cols * rows);
  std::vector<double> res(cols - 1, 0);
  std::vector<double> reference_res(cols - 1);

  for (int i = 0; i < rows; ++i) {
    matrix[i * cols + i] = 1;
    matrix[i * cols + rows] = i + 1;
  }
  std::iota(reference_res.begin(), reference_res.end(), 1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_EQ(MPIGaussHorizontalSequential.validation(), true);
  MPIGaussHorizontalSequential.pre_processing();
  MPIGaussHorizontalSequential.run();
  MPIGaussHorizontalSequential.post_processing();
  ASSERT_EQ(reference_res, res);
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_51x50) {
  const int cols = 51;
  const int rows = 50;

  std::vector<double> matrix(cols * rows);
  std::vector<double> res(cols - 1, 0);
  std::vector<double> reference_res(cols - 1);

  for (int i = 0; i < rows; ++i) {
    matrix[i * cols + i] = 1;
    matrix[i * cols + rows] = i + 1;
  }
  std::iota(reference_res.begin(), reference_res.end(), 1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_EQ(MPIGaussHorizontalSequential.validation(), true);
  MPIGaussHorizontalSequential.pre_processing();
  MPIGaussHorizontalSequential.run();
  MPIGaussHorizontalSequential.post_processing();
  ASSERT_EQ(reference_res, res);
}

TEST(shishkarev_a_gaussian_method_horizontal_strip_pattern_seq, test_101x100) {
  const int cols = 101;
  const int rows = 100;

  std::vector<double> matrix(cols * rows);
  std::vector<double> res(cols - 1, 0);
  std::vector<double> reference_res(cols - 1);

  for (int i = 0; i < rows; ++i) {
    matrix[i * cols + i] = 1;
    matrix[i * cols + rows] = i + 1;
  }
  std::iota(reference_res.begin(), reference_res.end(), 1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  shishkarev_a_gaussian_method_horizontal_strip_pattern_seq::MPIGaussHorizontalSequential MPIGaussHorizontalSequential(
      taskDataSeq);
  ASSERT_EQ(MPIGaussHorizontalSequential.validation(), true);
  MPIGaussHorizontalSequential.pre_processing();
  MPIGaussHorizontalSequential.run();
  MPIGaussHorizontalSequential.post_processing();
  ASSERT_EQ(reference_res, res);
}