#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/sedova_o_vertical_ribbon_scheme/include/ops_seq.hpp"

TEST(sedova_o_vertical_ribbon_scheme_seq, empty_matrix) {
  std::vector<int> matrix = {};
  std::vector<int> vector = {1, 2, 3};
  std::vector<int> result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_seq::Sequential TestSequential(taskDataSeq);
  EXPECT_FALSE(TestSequential.validation());
}

TEST(sedova_o_vertical_ribbon_scheme_seq, empty_vector) {
  std::vector<int> matrix = {1, 2, 4, 5};
  std::vector<int> vector = {};
  std::vector<int> result(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_seq::Sequential TestSequential(taskDataSeq);
  EXPECT_FALSE(TestSequential.validation());
}

TEST(sedova_o_vertical_ribbon_scheme_seq, empty_matrix_and_vector) {
  std::vector<int> matrix = {};
  std::vector<int> vector = {};
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_seq::Sequential TestSequential(taskDataSeq);
  EXPECT_FALSE(TestSequential.validation());
}

TEST(sedova_o_vertical_ribbon_scheme_seq, matrix_3x2) {
  std::vector<int> matrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> vector = {7, 8};
  std::vector<int> result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_seq::Sequential TestSequential(taskDataSeq);
  ASSERT_TRUE(TestSequential.validation());
  TestSequential.pre_processing();
  TestSequential.run();
  TestSequential.post_processing();

  std::vector<int> expected_result = {39, 54, 69};
  ASSERT_EQ(result, expected_result);
}

TEST(sedova_o_vertical_ribbon_scheme_seq, matrix_2x2) {
  std::vector<int> matrix = {1, 2, 3, 4};
  std::vector<int> vector = {5, 6};
  std::vector<int> result(2, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_seq::Sequential TestSequential(taskDataSeq);
  ASSERT_TRUE(TestSequential.validation());
  TestSequential.pre_processing();
  TestSequential.run();
  TestSequential.post_processing();

  std::vector<int> expected_result = {23, 34};
  ASSERT_EQ(result, expected_result);
}

TEST(sedova_o_vertical_ribbon_scheme_seq, matrix_5x2) {
  std::vector<int> matrix = {1, 3, 5, 4, 6, 7, 8, 2, 9, 10};
  std::vector<int> vector = {2, 6};
  std::vector<int> result(5, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_seq::Sequential TestSequential(taskDataSeq);
  ASSERT_TRUE(TestSequential.validation());
  TestSequential.pre_processing();
  TestSequential.run();
  TestSequential.post_processing();

  std::vector<int> expected_result = {44, 54, 22, 62, 72};
  ASSERT_EQ(result, expected_result);
}

TEST(sedova_o_vertical_ribbon_scheme_seq, matrix_5x1) {
  std::vector<int> matrix = {1, 3, 5, 4, 6};
  std::vector<int> vector = {2};
  std::vector<int> result(5, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_seq::Sequential TestSequential(taskDataSeq);
  ASSERT_TRUE(TestSequential.validation());
  TestSequential.pre_processing();
  TestSequential.run();
  TestSequential.post_processing();

  std::vector<int> expected_result = {2, 6, 10, 8, 12};
  ASSERT_EQ(result, expected_result);
}

TEST(sedova_o_vertical_ribbon_scheme_seq, false_validation1) {
  std::vector<int> matrix = {1, 2, 3, 4};
  std::vector<int> vector = {1, 2, 3};
  std::vector<int> result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_seq::Sequential TestSequential(taskDataSeq);
  EXPECT_FALSE(TestSequential.validation());
}

TEST(sedova_o_vertical_ribbon_scheme_seq, false_validation2) {
  std::vector<int> matrix = {1, 2, 3};
  std::vector<int> vector = {1, 2};
  std::vector<int> result(3, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(vector.data()));
  taskDataSeq->inputs_count.emplace_back(vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());

  sedova_o_vertical_ribbon_scheme_seq::Sequential TestSequential(taskDataSeq);
  EXPECT_FALSE(TestSequential.validation());
}