// Copyright 2023 Nesterov Alexander

#include <gtest/gtest.h>
#include <vector>
#include "seq/Shurygin_S_max_po_stolbam_matrix/include/ops_seq.hpp"

TEST(Shurygin_S_max_po_stolbam_matrix_seq, find_max_val_in_columns_10x10_matrix) {
  const int rows = 10;
  const int cols = 10;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential::generate_random_matrix(rows, cols);
  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  std::vector<int> v_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int j = 0; j < cols; j++) {
    int max_val = matrix_rnd[0][j];
    for (int i = 1; i < rows; i++) {
      if (matrix_rnd[i][j] > max_val) {
        max_val = matrix_rnd[i][j];
      }
    }
    ASSERT_EQ(v_res[j], max_val);
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_seq, find_max_val_in_columns_100x100_matrix) {
  const int rows = 100;
  const int cols = 100;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential::generate_random_matrix(rows, cols);
  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  std::vector<int> v_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int j = 0; j < cols; j++) {
    int max_val = matrix_rnd[0][j];
    for (int i = 1; i < rows; i++) {
      if (matrix_rnd[i][j] > max_val) {
        max_val = matrix_rnd[i][j];
      }
    }
    ASSERT_EQ(v_res[j], max_val);
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_seq, find_max_val_in_columns_100x500_matrix) {
  const int rows = 100;
  const int cols = 500;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential::generate_random_matrix(rows, cols);
  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  std::vector<int> v_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int j = 0; j < cols; j++) {
    int max_val = matrix_rnd[0][j];
    for (int i = 1; i < rows; i++) {
      if (matrix_rnd[i][j] > max_val) {
        max_val = matrix_rnd[i][j];
      }
    }
    ASSERT_EQ(v_res[j], max_val);
  }
}

TEST(Shurygin_S_max_po_stolbam_matrix_seq, find_max_val_in_columns_5000x5000_matrix) {
  const int rows = 5000;
  const int cols = 5000;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      Shurygin_S_max_po_stolbam_matrix_seq::TestTaskSequential::generate_random_matrix(rows, cols);
  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);
  std::vector<int> v_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int j = 0; j < cols; j++) {
    int max_val = matrix_rnd[0][j];
    for (int i = 1; i < rows; i++) {
      if (matrix_rnd[i][j] > max_val) {
        max_val = matrix_rnd[i][j];
      }
    }
    ASSERT_EQ(v_res[j], max_val);
  }
}
