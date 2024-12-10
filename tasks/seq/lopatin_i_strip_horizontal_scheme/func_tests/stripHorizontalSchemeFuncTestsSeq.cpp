#include <gtest/gtest.h>

#include "seq/lopatin_i_strip_horizontal_scheme/include/stripHorizontalSchemeHeaderSeq.hpp"

TEST(lopatin_i_strip_horizontal_scheme_seq, test_validation_empty_vector) {
  std::vector<int> inputMatrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> inputVector = {};
  std::vector<int> resultVector(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
  taskDataSequential->inputs_count.emplace_back(3);
  taskDataSequential->inputs_count.emplace_back(2);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSequential->inputs_count.emplace_back(0);
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataSequential->outputs_count.emplace_back(resultVector.size());

  lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(lopatin_i_strip_horizontal_scheme_seq, test_validation_empty_matrix) {
  std::vector<int> inputMatrix = {};
  std::vector<int> inputVector = {1, 2};
  std::vector<int> resultVector(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
  taskDataSequential->inputs_count.emplace_back(0);
  taskDataSequential->inputs_count.emplace_back(0);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSequential->inputs_count.emplace_back(inputVector.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataSequential->outputs_count.emplace_back(resultVector.size());

  lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(lopatin_i_strip_horizontal_scheme_seq, test_validation_small_vector) {
  std::vector<int> inputMatrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> inputVector = {1, 2};
  std::vector<int> resultVector(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
  taskDataSequential->inputs_count.emplace_back(3);
  taskDataSequential->inputs_count.emplace_back(2);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSequential->inputs_count.emplace_back(inputVector.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataSequential->outputs_count.emplace_back(resultVector.size());

  lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(lopatin_i_strip_horizontal_scheme_seq, test_validation_big_vector) {
  std::vector<int> inputMatrix = {1, 2, 3, 4, 5, 6};
  std::vector<int> inputVector = {1, 2, 3, 4};
  std::vector<int> resultVector(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
  taskDataSequential->inputs_count.emplace_back(3);
  taskDataSequential->inputs_count.emplace_back(2);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSequential->inputs_count.emplace_back(inputVector.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataSequential->outputs_count.emplace_back(resultVector.size());

  lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_FALSE(testTaskSequential.validation());
}

TEST(lopatin_i_strip_horizontal_scheme_seq, test_4x4_matrix) {
  std::vector<int> inputMatrix = {5, 12, 34, 6, 8, 2, 15, 4, 2, 0, 0, 4, 0, 6, 0, 8};
  std::vector<int> inputVector = {1, 2, 3, 4};
  std::vector<int> resultVector(4, 0);
  std::vector<int> expectedResult = {155, 73, 18, 44};

  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
  taskDataSequential->inputs_count.emplace_back(4);
  taskDataSequential->inputs_count.emplace_back(4);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSequential->inputs_count.emplace_back(inputVector.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataSequential->outputs_count.emplace_back(resultVector.size());

  lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(resultVector, expectedResult);
}

TEST(lopatin_i_strip_horizontal_scheme_seq, test_4x6_matrix) {
  std::vector<int> inputMatrix = {5, 12, 34, 6, 2, 6, 8, 2, 15, 4, 4, 8, 2, 0, 0, 4, 16, 32, 0, 6, 0, 8, 64, 128};
  std::vector<int> inputVector = {1, 2, 3, 4, 5, 6};
  std::vector<int> resultVector(4, 0);
  std::vector<int> expectedResult = {201, 141, 290, 1132};

  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
  taskDataSequential->inputs_count.emplace_back(6);
  taskDataSequential->inputs_count.emplace_back(4);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSequential->inputs_count.emplace_back(inputVector.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataSequential->outputs_count.emplace_back(resultVector.size());

  lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(resultVector, expectedResult);
}

TEST(lopatin_i_strip_horizontal_scheme_seq, test_6x4_matrix) {
  std::vector<int> inputMatrix = {5, 12, 34, 6, 8, 2, 15, 4, 2, 0, 0, 4, 0, 6, 0, 8, 20, 17, 20, 12, 2, 0, 2, 4};
  std::vector<int> inputVector = {1, 2, 3, 4};
  std::vector<int> resultVector(6, 0);
  std::vector<int> expectedResult = {155, 73, 18, 44, 162, 24};

  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputMatrix.data()));
  taskDataSequential->inputs_count.emplace_back(4);
  taskDataSequential->inputs_count.emplace_back(6);
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t *>(inputVector.data()));
  taskDataSequential->inputs_count.emplace_back(inputVector.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t *>(resultVector.data()));
  taskDataSequential->outputs_count.emplace_back(resultVector.size());

  lopatin_i_strip_horizontal_scheme_seq::TestTaskSequential testTaskSequential(taskDataSequential);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(resultVector, expectedResult);
}