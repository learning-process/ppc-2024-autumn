// Copyright 2024 Korobeinukov Arseny
#include <gtest/gtest.h>

#include "seq/korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B/include/ops_seq_korobeinikov.hpp"

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B, Test_1_without_negative_elemets) {
  // Create data
  std::vector<int> A = {3, 2, 1};
  std::vector<int> B = {1, 2, 3};
  int count_rows_A = 3;
  int count_cols_A = 1;
  int count_rows_B = 1;
  int count_cols_B = 3;

  std::vector<int> out(9, 0);
  std::vector<int> right_answer = {3, 6, 9, 2, 4, 6, 1, 2, 3};
  int count_rows_out = 3;
  int count_cols_out = 3;
  int count_rows_RA = 3;
  int count_cols_RA = 3;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out));
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  korobeinikov_a_test_task_seq_lab_02::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < right_answer.size(); i++) {
    ASSERT_EQ(right_answer[i], out[i]);
  }
  ASSERT_EQ(count_rows_out, count_rows_RA);
  ASSERT_EQ(count_cols_out, count_cols_RA);
}

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B, Test_2_with_negative_elemets) {
  // Create data
  std::vector<int> A = {-3, 2, -1};
  std::vector<int> B = {1, 2, 3};
  int count_rows_A = 3;
  int count_cols_A = 1;
  int count_rows_B = 1;
  int count_cols_B = 3;

  std::vector<int> out(9, 0);
  std::vector<int> right_answer = {-3, -6, -9, 2, 4, 6, -1, -2, -3};
  int count_rows_out = 3;
  int count_cols_out = 3;
  int count_rows_RA = 3;
  int count_cols_RA = 3;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out));
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  korobeinikov_a_test_task_seq_lab_02::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < right_answer.size(); i++) {
    ASSERT_EQ(right_answer[i], out[i]);
  }
  ASSERT_EQ(count_rows_out, count_rows_RA);
  ASSERT_EQ(count_cols_out, count_cols_RA);
}

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B, Test_3_only_zero) {
  // Create data
  std::vector<int> A = {0, 0, 0};
  std::vector<int> B = {0, 0, 0};
  int count_rows_A = 3;
  int count_cols_A = 1;
  int count_rows_B = 1;
  int count_cols_B = 3;

  std::vector<int> out(9, 0);
  std::vector<int> right_answer = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int count_rows_out = 3;
  int count_cols_out = 3;
  int count_rows_RA = 3;
  int count_cols_RA = 3;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out));
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  korobeinikov_a_test_task_seq_lab_02::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < right_answer.size(); i++) {
    ASSERT_EQ(right_answer[i], out[i]);
  }
  ASSERT_EQ(count_rows_out, count_rows_RA);
  ASSERT_EQ(count_cols_out, count_cols_RA);
}

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B, Test_4_validation_false_1) {
  // Create data
  std::vector<int> A = {0, 0, 0};
  std::vector<int> B = {0, 0, 0};
  int count_rows_A = 3;
  int count_cols_A = 1;
  int count_rows_B = 3;
  int count_cols_B = 1;

  std::vector<int> out(9, 0);
  int count_rows_out = 3;
  int count_cols_out = 3;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out));
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  korobeinikov_a_test_task_seq_lab_02::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(korobeinikov_matrix_multiplication_horizontal_scheme_A_vertical_scheme_B, Test_5_validation_false_2) {
  // Create data
  std::vector<int> A = {0, 0, 0};
  std::vector<int> B = {0, 0, 0};
  int count_rows_A = 3;
  int count_cols_A = 1;
  int count_rows_B = 1;
  int count_cols_B = 10;

  std::vector<int> out(9, 0);
  int count_rows_out = 3;
  int count_cols_out = 3;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_A));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_A));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(B.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_B));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_B));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows_out));
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count_cols_out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  korobeinikov_a_test_task_seq_lab_02::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}