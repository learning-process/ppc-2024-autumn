#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "seq/sadikov_I_sum_values_by_columns_matrix/include/sq_task.h"

TEST(sum_values_by_columns_matrix, check_validation1) {
  std::vector<int> in(144, 1);
  std::vector<int> in_index{12, 12};
  std::vector<int> out(12, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
}

TEST(sum_values_by_columns_matrix, check_validation2) {
  std::vector<int> in(144, 1);
  std::vector<int> in_index{12, 12};
  std::vector<int> out(15, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), false);
}

TEST(sum_values_by_columns_matrix, check_empty_matrix) {
  std::vector<int> in(0);
  std::vector<int> in_index{0, 0};
  std::vector<int> out(0, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[1]; ++i) {
    EXPECT_NEAR(out[i], 0.0, 1e-6);
  }
}

TEST(sum_values_by_columns_matrix, check_square_matrix) {
  std::vector<int> in(144, 1);
  std::vector<int> in_index{12, 12};
  std::vector<int> out(12, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[1]; ++i) {
    ASSERT_EQ(out[i], in_index[0]);
  }
}

TEST(sum_values_by_columns_matrix, check_square_matrix2) {
  std::vector<int> in(256, 1);
  std::vector<int> in_index{16, 16};
  std::vector<int> out(16, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[1]; ++i) {
    ASSERT_EQ(out[i], in_index[0]);
  }
}

TEST(sum_values_by_columns_matrix, check_square_matrix3) {
  std::vector<int> in(256, 1);
  std::vector<int> in_index{16, 16};
  std::vector<int> out(16, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[1]; ++i) {
    ASSERT_EQ(out[i], in_index[0]);
  }
}

TEST(sum_values_by_columns_matrix, check_rect_matrix1) {
  std::vector<int> in(200, 1);
  std::vector<int> in_index{10, 20};
  std::vector<int> out(20, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[1]; ++i) {
    ASSERT_EQ(out[i], in_index[0]);
  }
}

TEST(sum_values_by_columns_matrix, check_rect_matrix2) {
  std::vector<int> in(500, 1);
  std::vector<int> in_index{50, 10};
  std::vector<int> out(10, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[1]; ++i) {
    ASSERT_EQ(out[i], in_index[0]);
  }
}

TEST(sum_values_by_columns_matrix, check_rect_matrix3) {
  std::vector<int> in(10000, 1);
  std::vector<int> in_index{500, 20};
  std::vector<int> out(20, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[1]; ++i) {
    ASSERT_EQ(out[i], in_index[0]);
  }
}