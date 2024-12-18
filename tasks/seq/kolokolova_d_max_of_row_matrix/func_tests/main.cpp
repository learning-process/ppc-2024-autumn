// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/kolokolova_d_max_of_row_matrix/include/ops_seq.hpp"

TEST(kolokolova_d_max_of_row_matrix_seq, Test_Max_For_Rows1) {
  int count_rows = 3;
  // Создание данных
  std::vector<int> global_mat = {2, 5, 4, 7, 9, 3, 5, 6, 7, 9, 2, 4, 2, 5, 0};
  std::vector<int32_t> seq_max_vec(count_rows, 0);
  std::vector<int32_t> ans = {9, 9, 5};

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataSeq->inputs_count.emplace_back(global_mat.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_max_vec.data()));
  taskDataSeq->outputs_count.emplace_back(seq_max_vec.size());

  // Создание задачи
  kolokolova_d_max_of_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, seq_max_vec);
}

TEST(kolokolova_d_max_of_row_matrix_seq, Test_Max_For_Rows2) {
  int count_rows = 4;
  // Создание данных
  std::vector<int> global_mat = {1, 2, 6, 11, 3, 5, 6, 3, 5, 4, 10, 12, 20, 4, 8, 2};
  std::vector<int32_t> seq_max_vec(count_rows, 0);
  std::vector<int32_t> ans = {11, 6, 12, 20};

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataSeq->inputs_count.emplace_back(global_mat.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_max_vec.data()));
  taskDataSeq->outputs_count.emplace_back(seq_max_vec.size());

  // Создание задачи
  kolokolova_d_max_of_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, seq_max_vec);
}

TEST(kolokolova_d_max_of_row_matrix_seq, Test_Max_For_Rows3) {
  int count_rows = 5;
  // Создание данных
  std::vector<int> global_mat = {10, 4, 3, 9, 7, 9, 13, 4, 6, 7, 5, 9, 12, 4, 2, 1, 10, 9, 0, 8};
  std::vector<int32_t> seq_max_vec(count_rows, 0);
  std::vector<int32_t> ans = {10, 13, 9, 12, 10};

  // Создание TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_mat.data()));
  taskDataSeq->inputs_count.emplace_back(global_mat.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&count_rows));
  taskDataSeq->inputs_count.emplace_back((size_t)1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(seq_max_vec.data()));
  taskDataSeq->outputs_count.emplace_back(seq_max_vec.size());

  // Создание задачи
  kolokolova_d_max_of_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, seq_max_vec);
}
