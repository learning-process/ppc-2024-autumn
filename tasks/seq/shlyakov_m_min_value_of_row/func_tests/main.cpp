// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/shlyakov_m_min_value_of_row/include/ops_seq.hpp"

TEST(shlyakov_m_min_value_of_row_seq, test_validation) {
  const int sz_row = 100;
  const int sz_col = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  shlyakov_m_min_value_of_row_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> rand_matr =
      shlyakov_m_min_value_of_row_seq::TestTaskSequential::get_random_matr(sz_row, sz_col);

  for (auto& row : rand_matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(sz_row);
  taskDataSeq->inputs_count.emplace_back(sz_col);

  std::vector<int> result_vex(sz_row, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vex.data()));
  taskDataSeq->outputs_count.emplace_back(result_vex.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
}

TEST(shlyakov_m_min_value_of_row_seq, test_pre_processing) {
  const int sz_row = 100;
  const int sz_col = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  shlyakov_m_min_value_of_row_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> rand_matr =
      shlyakov_m_min_value_of_row_seq::TestTaskSequential::get_random_matr(sz_row, sz_col);

  for (auto& row : rand_matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(sz_row);
  taskDataSeq->inputs_count.emplace_back(sz_col);

  std::vector<int> result_vex(sz_row, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vex.data()));
  taskDataSeq->outputs_count.emplace_back(result_vex.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
}

TEST(shlyakov_m_min_value_of_row_seq, test_run) {
  const int sz_row = 100;
  const int sz_col = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  shlyakov_m_min_value_of_row_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> rand_matr =
      shlyakov_m_min_value_of_row_seq::TestTaskSequential::get_random_matr(sz_row, sz_col);

  for (auto& row : rand_matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(sz_row);
  taskDataSeq->inputs_count.emplace_back(sz_col);

  std::vector<int> result_vex(sz_row, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vex.data()));
  taskDataSeq->outputs_count.emplace_back(result_vex.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  ASSERT_TRUE(testTaskSequential.run());
}

TEST(shlyakov_m_min_value_of_row_seq, test_post_processing) {
  const int sz_row = 100;
  const int sz_col = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  shlyakov_m_min_value_of_row_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> rand_matr =
      shlyakov_m_min_value_of_row_seq::TestTaskSequential::get_random_matr(sz_row, sz_col);

  for (auto& row : rand_matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(sz_row);
  taskDataSeq->inputs_count.emplace_back(sz_col);

  std::vector<int> result_vex(sz_row, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vex.data()));
  taskDataSeq->outputs_count.emplace_back(result_vex.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  ASSERT_TRUE(testTaskSequential.post_processing());
}

TEST(shlyakov_m_min_value_of_row_seq, test_eq_result_square_matr) {
  const int sz_row = 100;
  const int sz_col = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  shlyakov_m_min_value_of_row_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> rand_matr =
      shlyakov_m_min_value_of_row_seq::TestTaskSequential::get_random_matr(sz_row, sz_col);

  for (auto& row : rand_matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(sz_row);
  taskDataSeq->inputs_count.emplace_back(sz_col);

  std::vector<int> result_vex(sz_row, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vex.data()));
  taskDataSeq->outputs_count.emplace_back(result_vex.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (int i = 0; i < sz_row; i++) {
    ASSERT_EQ(result_vex[i], INT_MIN);
  }
}

TEST(shlyakov_m_min_value_of_row_seq, test_eq_result_notsquare_matr) {
  const int sz_row = 150;
  const int sz_col = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  shlyakov_m_min_value_of_row_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> rand_matr =
      shlyakov_m_min_value_of_row_seq::TestTaskSequential::get_random_matr(sz_row, sz_col);

  for (auto& row : rand_matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(sz_row);
  taskDataSeq->inputs_count.emplace_back(sz_col);

  std::vector<int> result_vex(sz_row, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vex.data()));
  taskDataSeq->outputs_count.emplace_back(result_vex.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (int i = 0; i < sz_row; i++) {
    ASSERT_EQ(result_vex[i], INT_MIN);
  }
}

TEST(shlyakov_m_min_value_of_row_seq, test_validation_uncorrect_input) {
  const int sz_row = 0;
  const int sz_col = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  shlyakov_m_min_value_of_row_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> rand_matr =
      shlyakov_m_min_value_of_row_seq::TestTaskSequential::get_random_matr(sz_row, sz_col);

  for (auto& row : rand_matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(sz_row);
  taskDataSeq->inputs_count.emplace_back(sz_col);

  std::vector<int> result_vex(sz_row, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vex.data()));
  taskDataSeq->outputs_count.emplace_back(result_vex.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(shlyakov_m_min_value_of_row_seq, test_then_input_sz_not_eq_output_sz) {
  const int rows = 100;
  const int cols = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  shlyakov_m_min_value_of_row_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      shlyakov_m_min_value_of_row_seq::TestTaskSequential::get_random_matr(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> v_res(rows - 1, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}