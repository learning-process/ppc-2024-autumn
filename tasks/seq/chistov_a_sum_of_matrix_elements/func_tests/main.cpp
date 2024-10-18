#include <gtest/gtest.h>

#include <vector>

#include "seq/chistov_a_sum_of_matrix_elements/include/ops_seq.hpp"

TEST(chistov_a_sum_of_matrix_elements_seq, test_int_sum_sequential) {
  const int n = 3;
  const int m = 4;
  std::vector<int> global_matrix = chistov_a_sum_of_matrix_elements_seq::get_random_matrix_seq<int>(n, m);
  std::vector<int32_t> reference_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
  taskDataSeq->outputs_count.emplace_back(reference_sum.size());

  chistov_a_sum_of_matrix_elements_seq::TestTaskSequential<int> TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  int sum = chistov_a_sum_of_matrix_elements_seq::classic_way_seq(global_matrix, n, m);
  ASSERT_EQ(reference_sum[0], sum);
}

TEST(chistov_a_sum_of_matrix_elements_seq, test_double_sum_sequential) {
  const int n = 3;
  const int m = 4;
  std::vector<double> global_matrix = chistov_a_sum_of_matrix_elements_seq::get_random_matrix_seq<double>(n, m);
  std::vector<double> reference_sum(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
  taskDataSeq->outputs_count.emplace_back(reference_sum.size());

  chistov_a_sum_of_matrix_elements_seq::TestTaskSequential<double> TestTaskSequential(taskDataSeq);

  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();
  double sum = chistov_a_sum_of_matrix_elements_seq::classic_way_seq(global_matrix, n, m);

  ASSERT_NEAR(reference_sum[0], sum, 1e-6);
}

TEST(chistov_a_sum_of_matrix_elements_seq, test_sum_with_empty_matrix_sequential) {
  std::vector<int32_t> reference_sum(1, 0);
  std::vector<int> empty_matrix;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(empty_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(empty_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
  taskDataSeq->outputs_count.emplace_back(reference_sum.size());
  chistov_a_sum_of_matrix_elements_seq::TestTaskSequential<int> TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  ASSERT_EQ(reference_sum[0], 0);
}

TEST(chistov_a_sum_of_matrix_elements_seq, test_sum_with_single_element_matrix_sequential) {
  const int n = 1;
  const int m = 1;
  std::vector<int> global_matrix = chistov_a_sum_of_matrix_elements_seq::get_random_matrix_seq<int>(n, m);
  std::vector<int32_t> reference_sum(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_sum.data()));
  taskDataSeq->outputs_count.emplace_back(reference_sum.size());

  chistov_a_sum_of_matrix_elements_seq::TestTaskSequential<int> TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), true);
  TestTaskSequential.pre_processing();
  TestTaskSequential.run();
  TestTaskSequential.post_processing();

  int sum = chistov_a_sum_of_matrix_elements_seq::classic_way_seq(global_matrix, n, m);
  ASSERT_EQ(reference_sum[0], sum);
}

TEST(chistov_a_sum_of_matrix_elements_seq, returns_empty_matrix_when_small_n_or_m_sequential) {
  auto matrix1 = chistov_a_sum_of_matrix_elements_seq::get_random_matrix_seq<int>(0, 1);
  EXPECT_TRUE(matrix1.empty());
  auto matrix2 = chistov_a_sum_of_matrix_elements_seq::get_random_matrix_seq<int>(1, 0);
  EXPECT_TRUE(matrix2.empty());
}

TEST(chistov_a_sum_of_matrix_elements_seq, test_wrong_validation_sequential) {
  std::vector<int> global_matrix;
  std::vector<int32_t> global_sum(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  const int n = 3;
  const int m = 4;
  global_matrix = chistov_a_sum_of_matrix_elements_seq::get_random_matrix_seq<int>(n, m);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
  taskDataSeq->outputs_count.emplace_back(global_sum.size());
  chistov_a_sum_of_matrix_elements_seq::TestTaskSequential<int> TestTaskSequential(taskDataSeq);
  ASSERT_EQ(TestTaskSequential.validation(), false);
}
