#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <vector>

#include "seq/sadikov_I_gather/include/sq_task.h"

std::vector<int> getRandomVector(size_t size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(size);
  for (size_t i = 0; i < size; i++) {
    vec[i] = (gen() % 100) - 49;
  }
  return vec;
}

TEST(gather_seq, check_validation1) {
  std::vector<int> in(144, 1);
  std::vector<int> in_index{12, 12};
  std::vector<int> out(12, 0);
  std::shared_ptr<ppc::core::TaskData> taskData = sadikov_I_gather_seq::CreateTaskData(in, in_index, out);
  sadikov_I_gather_seq::SumByRowsTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
}

TEST(gather_seq, check_validation2) {
  std::vector<int> in(144, 1);
  std::vector<int> in_index{12, 12};
  std::vector<int> out(15, 0);
  std::shared_ptr<ppc::core::TaskData> taskData = sadikov_I_gather_seq::CreateTaskData(in, in_index, out);
  sadikov_I_gather_seq::SumByRowsTask sv(taskData);
  ASSERT_EQ(sv.validation(), false);
}

TEST(sum_values_by_columns_matrix, check_empty_matrix) {
  std::vector<int> in(0);
  std::vector<int> in_index{0, 0};
  std::vector<int> out(0, 0);
  std::shared_ptr<ppc::core::TaskData> taskData = sadikov_I_gather_seq::CreateTaskData(in, in_index, out);
  sadikov_I_gather_seq::SumByRowsTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[0]; ++i) {
    EXPECT_NEAR(out[i], 0.0, 1e-6);
  }
}

TEST(sum_values_by_columns_matrix, check_square_matrix) {
  std::vector<int> in(625, 1);
  std::vector<int> in_index{25, 25};
  std::vector<int> out(25, 0);
  std::shared_ptr<ppc::core::TaskData> taskData = sadikov_I_gather_seq::CreateTaskData(in, in_index, out);
  sadikov_I_gather_seq::SumByRowsTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[0]; ++i) {
    ASSERT_EQ(out[i], in_index[1]);
  }
}

TEST(sum_values_by_columns_matrix, check_square_matrix2) {
  std::vector<int> in(900, 1);
  std::vector<int> in_index{30, 30};
  std::vector<int> out(30, 0);
  std::shared_ptr<ppc::core::TaskData> taskData = sadikov_I_gather_seq::CreateTaskData(in, in_index, out);
  sadikov_I_gather_seq::SumByRowsTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[0]; ++i) {
    ASSERT_EQ(out[i], in_index[1]);
  }
}

TEST(sum_values_by_columns_matrix, check_square_matrix3) {
  std::vector<int> in(256, 1);
  std::vector<int> in_index{16, 16};
  std::vector<int> out(16, 0);
  std::shared_ptr<ppc::core::TaskData> taskData = sadikov_I_gather_seq::CreateTaskData(in, in_index, out);
  sadikov_I_gather_seq::SumByRowsTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[0]; ++i) {
    ASSERT_EQ(out[i], in_index[1]);
  }
}

TEST(sum_values_by_columns_matrix, check_rect_matrix1) {
  std::vector<int> in(500, 1);
  std::vector<int> in_index{50, 10};
  std::vector<int> out(50, 0);
  std::shared_ptr<ppc::core::TaskData> taskData = sadikov_I_gather_seq::CreateTaskData(in, in_index, out);
  sadikov_I_gather_seq::SumByRowsTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[0]; ++i) {
    ASSERT_EQ(out[i], in_index[1]);
  }
}

TEST(sum_values_by_columns_matrix, check_rect_matrix2) {
  std::vector<int> in(720, 1);
  std::vector<int> in_index{80, 9};
  std::vector<int> out(80, 0);
  std::shared_ptr<ppc::core::TaskData> taskData = sadikov_I_gather_seq::CreateTaskData(in, in_index, out);
  sadikov_I_gather_seq::SumByRowsTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  for (int i = 0; i < in_index[0]; ++i) {
    ASSERT_EQ(out[i], in_index[1]);
  }
}

TEST(sum_values_by_columns_matrix, check_rect_matrix3) {
  std::vector<int> in_index{50, 200};
  std::vector<int> out(50, 0);
  std::vector<int> in = getRandomVector(in_index[0] * in_index[1]);
  std::shared_ptr<ppc::core::TaskData> taskData = sadikov_I_gather_seq::CreateTaskData(in, in_index, out);
  sadikov_I_gather_seq::SumByRowsTask sv(taskData);
  ASSERT_EQ(sv.validation(), true);
  sv.pre_processing();
  sv.run();
  sv.post_processing();
  std::vector<int> check_answer(in_index[0], 0);
  for (int i = 0; i < in_index[0]; ++i) {
    for (int j = 0; j < in_index[1]; ++j) {
      check_answer[i] += in[i * in_index[1] + j];
    }
  }
  for (int i = 0; i < in_index[0]; ++i) {
    ASSERT_EQ(out[i], check_answer[i]);
  }
}