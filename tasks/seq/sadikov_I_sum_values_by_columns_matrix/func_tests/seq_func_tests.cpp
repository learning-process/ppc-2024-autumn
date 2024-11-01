#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "seq/sadikov_I_sum_values_by_columns_matrix/include/sq_task.h"

TEST(sum_values_by_columns_matrix, check_validation1) {
  std::vector<double> in(144, 1);
  std::vector<size_t> in_index{12, 12};
  std::vector<double> out(12, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
}

TEST(sum_values_by_columns_matrix, check_validation2) {
  std::vector<double> in(144, 1);
  std::vector<size_t> in_index{12, 12};
  std::vector<double> out(15, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), false);
}

TEST(sum_values_by_columns_matrix, check_empty_matrix) {
  std::vector<double> in(0);
  std::vector<size_t> in_index{0, 0};
  std::vector<double> out(0, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (size_t i = 0; i < in_index[1]; ++i) {
    EXPECT_NEAR(out[i], 0.0, 1e-6);
  }
}

TEST(sum_values_by_columns_matrix, check_square_matrix) {
  std::vector<double> in(144, 1.0 / 12.0);
  std::vector<size_t> in_index{12, 12};
  std::vector<double> out(12, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (size_t i = 0; i < in_index[1]; ++i) {
    EXPECT_NEAR(out[i], 1.0, 1e-6);
  }
}

TEST(sum_values_by_columns_matrix, check_square_matrix2) {
  std::vector<double> in(256, 1.0 / 16.0);
  std::vector<size_t> in_index{16, 16};
  std::vector<double> out(16, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (size_t i = 0; i < in_index[1]; ++i) {
    EXPECT_NEAR(out[i], 1.0, 1e-6);
  }
}

TEST(sum_values_by_columns_matrix, check_square_matrix3) {
  std::vector<double> in(256);
  std::vector<size_t> in_index{16, 16};
  std::vector<double> out(16, 0);
  in = sadikov_I_Sum_values_by_columns_matrix_seq::Randvector(256);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (size_t i = 0; i < in_index[1]; ++i) {
    ASSERT_LE(out[i], 1.0);
  }
}

TEST(sum_values_by_columns_matrix, check_rect_matrix1) {
  std::vector<double> in(200, 1.0 / 10.0);
  std::vector<size_t> in_index{10, 20};
  std::vector<double> out(20, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (size_t i = 0; i < in_index[1]; ++i) {
    EXPECT_NEAR(out[i], 1.0, 1e-6);
  }
}

TEST(sum_values_by_columns_matrix, check_rect_matrix2) {
  std::vector<double> in(500, 1.0 / 50.0);
  std::vector<size_t> in_index{50, 10};
  std::vector<double> out(10, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (size_t i = 0; i < in_index[1]; ++i) {
    EXPECT_NEAR(out[i], 1.0, 1e-6);
  }
}

TEST(sum_values_by_columns_matrix, check_rect_matrix3) {
  std::vector<double> in(10000, 1.0 / 500.0);
  std::vector<size_t> in_index{500, 20};
  std::vector<double> out(20, 0);
  std::shared_ptr<ppc::core::TaskData> taskData =
      sadikov_I_Sum_values_by_columns_matrix_seq::CreateTaskData(in, in_index, out);
  sadikov_I_Sum_values_by_columns_matrix_seq::MatrixTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (size_t i = 0; i < in_index[1]; ++i) {
    EXPECT_NEAR(out[i], 1.0, 1e-6);
  }
}