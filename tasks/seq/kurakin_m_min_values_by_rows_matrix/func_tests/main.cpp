// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/kurakin_m_min_values_by_rows_matrix/include/ops_seq.hpp"

TEST(kurakin_m_min_values_by_rows_matrix_seq, Test_Min1) {
  int count_rows = 3;
  int size_rows = 5;

  std::vector<int> global_mat = {1, 5, 3, 7, 9, 3, 4, 6, 7, 9, 2, 4, 2, 5, 0};
  // Create data
  std::vector<int32_t> seq_min_vec(count_rows, 0);
  std::vector<int32_t> ans = {1, 3, 0};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataSeq->inputs_count.emplace_back(global_mat.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_min_vec.data()));
  taskDataSeq->outputs_count.emplace_back(seq_min_vec.size());

  // Create Task
  kurakin_m_min_values_by_rows_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq, count_rows, size_rows);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, seq_min_vec);
}

TEST(kurakin_m_min_values_by_rows_matrix_seq, Test_Min2) {
  int count_rows = 3;
  int size_rows = 6;

  std::vector<int> global_mat = {10, 5, 3, 9, 7, 9, 13, 4, 6, 7, 7, 9, 12, 4, 2, 5, 10, 9};

  // Create data
  std::vector<int32_t> seq_min_vec(count_rows, 0);
  std::vector<int32_t> ans = {3, 4, 2};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataSeq->inputs_count.emplace_back(global_mat.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_min_vec.data()));
  taskDataSeq->outputs_count.emplace_back(seq_min_vec.size());

  // Create Task
  kurakin_m_min_values_by_rows_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq, count_rows, size_rows);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, seq_min_vec);
}

TEST(kurakin_m_min_values_by_rows_matrix_seq, Test_Min3) {
  int count_rows = 4;
  int size_rows = 5;

  std::vector<int> global_mat = {10, 5, 3, 9, 7, 9, 13, 4, 6, 7, 7, 9, 12, 4, 2, 5, 10, 9, 5, 8};
  // Create data
  std::vector<int32_t> seq_min_vec(count_rows, 0);
  std::vector<int32_t> ans = {3, 4, 2, 5};
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataSeq->inputs_count.emplace_back(global_mat.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_min_vec.data()));
  taskDataSeq->outputs_count.emplace_back(seq_min_vec.size());

  // Create Task
  kurakin_m_min_values_by_rows_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq, count_rows, size_rows);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, seq_min_vec);
}

int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
