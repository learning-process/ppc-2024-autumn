// Copyright 2024 Korobeinikov Arseny
#include <gtest/gtest.h>

#include "seq/korobeinikov_dijkstras_algorithm/include/ops_seq_korobeinikov.hpp"

TEST(korobeinikov_dijkstras_algorithm_seq, Test_1_const_numbers) {
  // Create data
  std::vector<int> values = {10, 30, 100, 50, 10, 20, 60};
  std::vector<int> col = {1, 3, 4, 2, 4, 2, 4};
  std::vector<int> RowIndex = {0, 3, 4, 5, 7, 7};

  int size = 5;
  int sv = 0;

  std::vector<int> out(size, 0);
  std::vector<int> right_answer = {0, 10, 50, 30, 60};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
  taskDataSeq->inputs_count.emplace_back(col.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
  taskDataSeq->inputs_count.emplace_back(RowIndex.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  korobeinikov_a_test_task_seq_lab_03::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (size_t i = 0; i < right_answer.size(); i++) {
    ASSERT_EQ(right_answer[i], out[i]);
  }
}

TEST(korobeinikov_dijkstras_algorithm_seq, Test_2_emty_graph) {
  // Create data
  std::vector<int> values;
  std::vector<int> col;
  std::vector<int> RowIndex;

  int size = 0;
  int sv = 0;

  std::vector<int> out(size, 0);
  std::vector<int> right_answer = {0, 10, 50, 30, 60};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
  taskDataSeq->inputs_count.emplace_back(col.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
  taskDataSeq->inputs_count.emplace_back(RowIndex.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  korobeinikov_a_test_task_seq_lab_03::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(korobeinikov_dijkstras_algorithm_seq, Test_2_validation_false_sv_greater_then_size) {
  // Create data
  std::vector<int> values;
  std::vector<int> col;
  std::vector<int> RowIndex;

  int size = 5;
  int sv = 10;

  std::vector<int> out(size, 0);
  std::vector<int> right_answer = {0, 10, 50, 30, 60};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
  taskDataSeq->inputs_count.emplace_back(col.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
  taskDataSeq->inputs_count.emplace_back(RowIndex.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  korobeinikov_a_test_task_seq_lab_03::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(korobeinikov_dijkstras_algorithm_seq, Test_2_validation_false_incorrect_size) {
  // Create data
  std::vector<int> values;
  std::vector<int> col;
  std::vector<int> RowIndex;

  int size = 10;
  int sv = 0;

  std::vector<int> out(size, 0);
  std::vector<int> right_answer = {0, 10, 50, 30, 60};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(values.data()));
  taskDataSeq->inputs_count.emplace_back(values.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(col.data()));
  taskDataSeq->inputs_count.emplace_back(col.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(RowIndex.data()));
  taskDataSeq->inputs_count.emplace_back(RowIndex.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sv));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  korobeinikov_a_test_task_seq_lab_03::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}