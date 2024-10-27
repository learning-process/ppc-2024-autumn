// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/kondratev_ya_max_col_matrix/include/ops_seq.hpp"

void runTask(ppc::core::Task& task) {
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
}

void fillTaskData(std::shared_ptr<ppc::core::TaskData> taskData, uint32_t row, uint32_t col, auto& mtrx, auto& res) {
  for (auto& mtrxRow : mtrx) taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(mtrxRow.data()));
  taskData->inputs_count.emplace_back(row);
  taskData->inputs_count.emplace_back(col);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskData->outputs_count.emplace_back(res.size());
}

TEST(kondratev_ya_max_col_matrix_seq, test_1) {
  uint32_t row = 100;
  uint32_t col = 100;
  int32_t ref_val = INT_MAX;

  std::vector<int32_t> res(col);
  std::vector<int32_t> ref(col, ref_val);
  std::vector<std::vector<int32_t>> mtrx = kondratev_ya_max_col_matrix_seq::getRandomMatrix(row, col);
  kondratev_ya_max_col_matrix_seq::insertRefValue(mtrx, ref_val);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  fillTaskData(taskDataSeq, row, col, mtrx, res);

  kondratev_ya_max_col_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  runTask(testTaskSequential);

  ASSERT_EQ(res, ref);
}

TEST(kondratev_ya_max_col_matrix_seq, test_2) {
  uint32_t row = 1000;
  uint32_t col = 50;
  int32_t ref_val = INT_MAX;

  std::vector<int32_t> res(col);
  std::vector<int32_t> ref(col, ref_val);
  std::vector<std::vector<int32_t>> mtrx = kondratev_ya_max_col_matrix_seq::getRandomMatrix(row, col);
  kondratev_ya_max_col_matrix_seq::insertRefValue(mtrx, ref_val);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  fillTaskData(taskDataSeq, row, col, mtrx, res);

  kondratev_ya_max_col_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  runTask(testTaskSequential);

  ASSERT_EQ(res, ref);
}

TEST(kondratev_ya_max_col_matrix_seq, test_3) {
  uint32_t row = 500;
  uint32_t col = 1000;
  int32_t ref_val = INT_MAX;
  std::vector<int32_t> res(col);
  std::vector<int32_t> ref(col, ref_val);
  std::vector<std::vector<int32_t>> mtrx = kondratev_ya_max_col_matrix_seq::getRandomMatrix(row, col);
  kondratev_ya_max_col_matrix_seq::insertRefValue(mtrx, ref_val);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  fillTaskData(taskDataSeq, row, col, mtrx, res);

  kondratev_ya_max_col_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  runTask(testTaskSequential);

  ASSERT_EQ(res, ref);
}
