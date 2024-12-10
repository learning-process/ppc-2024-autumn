// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <numeric>
#include <random>
#include <vector>

#include "seq/malyshev_a_sum_rows_matrix/include/ops_seq.hpp"

namespace malyshev_a_sum_rows_matrix_test_function {

std::vector<std::vector<int32_t>> getRandomData(uint32_t rows, uint32_t cols) {
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

}  // namespace malyshev_a_sum_rows_matrix_test_function

TEST(malyshev_a_sum_rows_matrix_seq, rectangular_matrix_stretched_horizontally_7x17) {
  uint32_t rows = 7;
  uint32_t cols = 17;

  std::vector<int32_t> seqSum(rows);
  std::vector<std::vector<int32_t>> data(rows, std::vector(cols, 1));

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_seq::TestTaskSequential taskSeq(taskDataSeq);

  for (auto &row : data) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
  taskDataSeq->outputs_count.push_back(seqSum.size());

  ASSERT_TRUE(taskSeq.validation());
  ASSERT_TRUE(taskSeq.pre_processing());
  ASSERT_TRUE(taskSeq.run());
  ASSERT_TRUE(taskSeq.post_processing());

  for (uint32_t i = 0; i < seqSum.size(); i++) {
    ASSERT_EQ(seqSum[i], (int32_t)cols);
  }
}

TEST(malyshev_a_sum_rows_matrix_seq, rectangular_matrix_stretched_verticaly_100x75) {
  uint32_t rows = 100;
  uint32_t cols = 75;

  std::vector<int32_t> seqSum(rows);
  std::vector<std::vector<int32_t>> randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_seq::TestTaskSequential taskSeq(taskDataSeq);

  for (auto &row : randomData) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
  taskDataSeq->outputs_count.push_back(seqSum.size());

  ASSERT_TRUE(taskSeq.validation());
  ASSERT_TRUE(taskSeq.pre_processing());
  ASSERT_TRUE(taskSeq.run());
  ASSERT_TRUE(taskSeq.post_processing());

  for (uint32_t i = 0; i < seqSum.size(); i++) {
    ASSERT_EQ(seqSum[i], std::accumulate(randomData[i].begin(), randomData[i].end(), 0));
  }
}

TEST(malyshev_a_sum_rows_matrix_seq, squere_matrix_100x100) {
  uint32_t rows = 100;
  uint32_t cols = 100;

  std::vector<int32_t> seqSum(rows);
  std::vector<std::vector<int32_t>> randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_seq::TestTaskSequential taskSeq(taskDataSeq);

  for (auto &row : randomData) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
  taskDataSeq->outputs_count.push_back(seqSum.size());

  ASSERT_TRUE(taskSeq.validation());
  ASSERT_TRUE(taskSeq.pre_processing());
  ASSERT_TRUE(taskSeq.run());
  ASSERT_TRUE(taskSeq.post_processing());

  for (uint32_t i = 0; i < seqSum.size(); i++) {
    ASSERT_EQ(seqSum[i], std::accumulate(randomData[i].begin(), randomData[i].end(), 0));
  }
}

TEST(malyshev_a_sum_rows_matrix_seq, matrix_1x1) {
  uint32_t rows = 1;
  uint32_t cols = 1;

  std::vector<int32_t> seqSum(rows);
  std::vector<std::vector<int32_t>> randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_seq::TestTaskSequential taskSeq(taskDataSeq);

  for (auto &row : randomData) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
  taskDataSeq->outputs_count.push_back(seqSum.size());

  ASSERT_TRUE(taskSeq.validation());
  ASSERT_TRUE(taskSeq.pre_processing());
  ASSERT_TRUE(taskSeq.run());
  ASSERT_TRUE(taskSeq.post_processing());

  for (uint32_t i = 0; i < seqSum.size(); i++) {
    ASSERT_EQ(seqSum[i], std::accumulate(randomData[i].begin(), randomData[i].end(), 0));
  }
}

TEST(malyshev_a_sum_rows_matrix_seq, test_validation) {
  uint32_t rows = 7;
  uint32_t cols = 17;

  std::vector<int32_t> seqSum(rows);
  std::vector<std::vector<int32_t>> randomData = malyshev_a_sum_rows_matrix_test_function::getRandomData(rows, cols);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_sum_rows_matrix_seq::TestTaskSequential taskSeq(taskDataSeq);

  for (auto &row : randomData) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(row.data()));
  }

  taskDataSeq->inputs_count.push_back(rows);
  taskDataSeq->inputs_count.push_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seqSum.data()));
  taskDataSeq->outputs_count.push_back(0);

  ASSERT_FALSE(taskSeq.validation());
}