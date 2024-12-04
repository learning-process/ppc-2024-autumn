// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include <vector>

#include "seq/korobeinikov_a_max_elements_in_rows_of_matrix/include/ops_seq_korobeinikov.hpp"

TEST(max_elements_in_rows_of_matrix_seq, Test_1_without_negative_max_elemet) {
  // Create data
  int count_rows = 4;  // not const, because reinterpret_cast does not work with const
  std::vector<int> matrix{3, 17, 5, -1, 2, -3, 11, 12, 13, -7, 4, 9};

  std::vector<int> seq_res(count_rows, 0);
  std::vector<int> right_answer = {17, 2, 13, 9};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  korobeinikov_a_test_task_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(right_answer, seq_res);
}

TEST(max_elements_in_rows_of_matrix_seq, Test_2_with_negative_max_elemet) {
  // Create data
  int count_rows = 4;  // not const, because reinterpret_cast does not work with const
  std::vector<int> matrix{3, 7, 5, -6, -10, -8, 15, 12, 21, -7, 0, 9};

  std::vector<int> seq_res(count_rows, 0);
  std::vector<int> right_answer = {7, -6, 21, 9};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  korobeinikov_a_test_task_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(right_answer, seq_res);
}

TEST(max_elements_in_rows_of_matrix_seq, Test_3_only_zero) {
  // Create data
  int count_rows = 2;  // not const, because reinterpret_cast does not work with const
  std::vector<int> matrix{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  std::vector<int> seq_res(count_rows, 0);
  std::vector<int> right_answer = {0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  korobeinikov_a_test_task_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(right_answer, seq_res);
}

TEST(max_elements_in_rows_of_matrix_seq, Test_4_empty_matrix) {
  // Create data
  int count_rows = 0;  // not const, because reinterpret_cast does not work with const
  std::vector<int> matrix;

  std::vector<int> seq_res(count_rows, 0);
  std::vector<int> right_answer(count_rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  korobeinikov_a_test_task_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(right_answer, seq_res);
}

TEST(max_elements_in_rows_of_matrix_seq, Test_5_Unequal_number_of_elements_in_rows_exeption) {
  // Create data
  int count_rows = 2;  // not const, because reinterpret_cast does not work with const
  std::vector<int> matrix{1, 2, 3, 4, 5};

  std::vector<int> seq_res(count_rows, 0);
  std::vector<int> right_answer = {0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  korobeinikov_a_test_task_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(max_elements_in_rows_of_matrix_seq,
     Test_6_number_of_elements_in_the_output_is_not_equal_to_number_of_rows_exeption) {
  // Create data
  int count_rows = 2;  // not const, because reinterpret_cast does not work with const
  std::vector<int> matrix{1, 2, 3, 4, 5};

  std::vector<int> seq_res(count_rows, 0);
  std::vector<int> right_answer = {0, 0};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_res.data()));
  taskDataSeq->outputs_count.emplace_back(seq_res.size());

  // Create Task
  korobeinikov_a_test_task_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}
