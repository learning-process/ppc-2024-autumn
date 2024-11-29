#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "seq/malyshev_v_lent_horizontal/include/ops_seq.hpp"

namespace malyshev_lent_horizontal {

std::vector<std::vector<int32_t>> generateRandomMatrix(uint32_t rows, uint32_t cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<std::vector<int32_t>> data(rows, std::vector<int32_t>(cols));

  for (auto &row : data) {
    for (auto &el : row) {
      el = -200 + gen() % (300 + 200 + 1);
    }
  }

  return data;
}

std::vector<int32_t> generateRandomVector(uint32_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int32_t> data(size);

  for (auto &el : data) {
    el = -200 + gen() % (300 + 200 + 1);
  }

  return data;
}

}  // namespace malyshev_lent_horizontal

TEST(malyshev_lent_horizontal, test_vertical_stretched_matrix_100x75) {
  uint32_t rows = 100;
  uint32_t cols = 75;

  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> seqResult;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

  randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
  randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
  seqResult.resize(rows);

  for (auto &row : randomMatrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
  taskDataSeq->outputs_count.push_back(seqResult.size());

  ASSERT_TRUE(taskSeq.validation());
  ASSERT_TRUE(taskSeq.pre_processing());
  ASSERT_TRUE(taskSeq.run());
  ASSERT_TRUE(taskSeq.post_processing());
}

TEST(malyshev_lent_horizontal, test_horizontal_stretched_matrix_7x17) {
  uint32_t rows = 7;
  uint32_t cols = 17;

  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> seqResult;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

  randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
  randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
  seqResult.resize(rows);

  for (auto &row : randomMatrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
  taskDataSeq->outputs_count.push_back(seqResult.size());

  ASSERT_TRUE(taskSeq.validation());
  ASSERT_TRUE(taskSeq.pre_processing());
  ASSERT_TRUE(taskSeq.run());
  ASSERT_TRUE(taskSeq.post_processing());
}

TEST(malyshev_lent_horizontal, test_square_matrix_100x100) {
  uint32_t rows = 100;
  uint32_t cols = 100;

  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> seqResult;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

  randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
  randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
  seqResult.resize(rows);

  for (auto &row : randomMatrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
  taskDataSeq->outputs_count.push_back(seqResult.size());

  ASSERT_TRUE(taskSeq.validation());
  ASSERT_TRUE(taskSeq.pre_processing());
  ASSERT_TRUE(taskSeq.run());
  ASSERT_TRUE(taskSeq.post_processing());
}

TEST(malyshev_lent_horizontal, test_single_element_matrix_1x1) {
  uint32_t rows = 1;
  uint32_t cols = 1;

  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> seqResult;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

  randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
  randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
  seqResult.resize(rows);

  for (auto &row : randomMatrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
  taskDataSeq->outputs_count.push_back(seqResult.size());

  ASSERT_TRUE(taskSeq.validation());
  ASSERT_TRUE(taskSeq.pre_processing());
  ASSERT_TRUE(taskSeq.run());
  ASSERT_TRUE(taskSeq.post_processing());
}

TEST(malyshev_lent_horizontal, test_validation_failure) {
  uint32_t rows = 7;
  uint32_t cols = 17;

  std::vector<std::vector<int32_t>> randomMatrix;
  std::vector<int32_t> randomVector;
  std::vector<int32_t> seqResult;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

  randomMatrix = malyshev_lent_horizontal::generateRandomMatrix(rows, cols);
  randomVector = malyshev_lent_horizontal::generateRandomVector(cols);
  seqResult.resize(rows);

  for (auto &row : randomMatrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(randomVector.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
  taskDataSeq->outputs_count.push_back(0);

  ASSERT_FALSE(taskSeq.validation());
}

TEST(malyshev_lent_horizontal, test_zero_values_sequential) {
  uint32_t rows = 3;
  uint32_t cols = 3;

  std::vector<std::vector<int32_t>> zeroMatrix(rows, std::vector<int32_t>(cols, 0));
  std::vector<int32_t> zeroVector(cols, 0);
  std::vector<int32_t> seqResult(rows, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_lent_horizontal::TestTaskSequential taskSeq(taskDataSeq);

  for (auto &row : zeroMatrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(zeroVector.data()));
  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqResult.data()));
  taskDataSeq->outputs_count.push_back(seqResult.size());

  ASSERT_TRUE(taskSeq.validation());
  ASSERT_TRUE(taskSeq.pre_processing());
  ASSERT_TRUE(taskSeq.run());
  ASSERT_TRUE(taskSeq.post_processing());

  for (uint32_t i = 0; i < seqResult.size(); i++) {
    ASSERT_EQ(seqResult[i], 0);
  }
}