#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/shulpin_i_strip_hA_vB/include/strip_hA_vB.hpp"

TEST(shulpin_strip_scheme_A_B, matrix_1x1) {
  int rows_a = 1;
  int cols_a = 1;
  int rows_b = 1;
  int cols_b = 1;

  std::vector<int> A = {2};
  std::vector<int> B = {3};
  std::vector<int> exp_res = {6};
  std::vector<int> C(rows_a * cols_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  shulpin_strip_scheme_A_B::Matrix_hA_vB_seq matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (int i = 0; i < C.size(); ++i) {
    ASSERT_EQ(C[i], exp_res[i]);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_3x3) {
  int rows_a = 3;
  int cols_a = 3;
  int rows_b = 3;
  int cols_b = 3;

  std::vector<int> A = {1, 0, 2, 0, 3, 0, 4, 0, 5};
  std::vector<int> B = {1, 2, 3, 0, 4, 0, 5, 6, 7};
  std::vector<int> exp_res = {11, 14, 17, 0, 12, 0, 29, 38, 47};
  std::vector<int> C(rows_a * cols_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  shulpin_strip_scheme_A_B::Matrix_hA_vB_seq matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (int i = 0; i < C.size(); ++i) {
    ASSERT_EQ(C[i], exp_res[i]);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_2x2) {
  int rows_a = 2;
  int cols_a = 2;
  int rows_b = 2;
  int cols_b = 2;

  std::vector<int> A = {1, 2, 3, 4};
  std::vector<int> B = {5, 6, 7, 8};
  std::vector<int> exp_res = {19, 22, 43, 50};
  std::vector<int> C(rows_a * cols_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  shulpin_strip_scheme_A_B::Matrix_hA_vB_seq matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (int i = 0; i < C.size(); ++i) {
    ASSERT_EQ(C[i], exp_res[i]);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_5x5) {
  int rows_a = 5;
  int cols_a = 5;
  int rows_b = 5;
  int cols_b = 5;

  std::vector<int> A = {1, 2, 3, 4, 5, 6, 7 ,8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
  std::vector<int> B = {25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> exp_res = {175, 160, 145, 130, 115, 550, 510, 470, 430, 390, 925, 860, 795, 730, 665, 1300, 1210, 1120, 1030, 940, 1675, 1560, 1445, 1330, 1215};
  std::vector<int> C(rows_a * cols_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  shulpin_strip_scheme_A_B::Matrix_hA_vB_seq matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (int i = 0; i < C.size(); ++i) {
    ASSERT_EQ(C[i], exp_res[i]);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_3x2_2x4) {
  int rows_a = 3;
  int cols_a = 2;
  int rows_b = 2;
  int cols_b = 4;

  std::vector<int> A = {1, 2, 3, 4, 5, 6};
  std::vector<int> B = {7, 8, 9, 10, 11, 12, 13, 14};
  std::vector<int> exp_res = {29, 32, 35, 38, 65, 72, 79, 86, 101, 112, 123, 134};
  std::vector<int> C(rows_a * cols_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  shulpin_strip_scheme_A_B::Matrix_hA_vB_seq matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (int i = 0; i < C.size(); ++i) {
    ASSERT_EQ(C[i], exp_res[i]);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_4x3_3x2) {
  int rows_a = 4;
  int cols_a = 3;
  int rows_b = 3;
  int cols_b = 2;

  std::vector<int> A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> B = {1, 4, 2, 5, 3, 6};
  std::vector<int> exp_res = {14, 32, 32, 77, 50, 122, 68, 167};
  std::vector<int> C(rows_a * cols_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  shulpin_strip_scheme_A_B::Matrix_hA_vB_seq matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (int i = 0; i < C.size(); ++i) {
    ASSERT_EQ(C[i], exp_res[i]);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_1x5_5x1) {
  int rows_a = 1;
  int cols_a = 5;
  int rows_b = 5;
  int cols_b = 1;

  std::vector<int> A = {1, 2, 3, 4, 5};
  std::vector<int> B = {5, 4, 3, 2, 1};
  std::vector<int> exp_res = {35};
  std::vector<int> C(rows_a * cols_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  shulpin_strip_scheme_A_B::Matrix_hA_vB_seq matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (int i = 0; i < C.size(); ++i) {
    ASSERT_EQ(C[i], exp_res[i]);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_5x1_1x5) {
  int rows_a = 5;
  int cols_a = 1;
  int rows_b = 1;
  int cols_b = 5;

  std::vector<int> A = {1, 2, 3, 4, 5};
  std::vector<int> B = {1, 2, 3, 4, 5};
  std::vector<int> exp_res = {1, 2, 3, 4, 5, 2, 4, 6, 8, 10, 3, 6, 9, 12, 15, 4, 8, 12, 16, 20, 5, 10, 15, 20, 25};
  std::vector<int> C(rows_a * cols_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  shulpin_strip_scheme_A_B::Matrix_hA_vB_seq matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (int i = 0; i < C.size(); ++i) {
    ASSERT_EQ(C[i], exp_res[i]);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_2x4_4x3) {
  int rows_a = 2;
  int cols_a = 4;
  int rows_b = 4;
  int cols_b = 2;

  std::vector<int> A = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> B = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> exp_res = {50, 60, 114, 140};
  std::vector<int> C(rows_a * cols_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  shulpin_strip_scheme_A_B::Matrix_hA_vB_seq matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (int i = 0; i < C.size(); ++i) {
    ASSERT_EQ(C[i], exp_res[i]);
  }
}

TEST(shulpin_strip_scheme_A_B, matrix_4x2_2x3) {
  int rows_a = 4;
  int cols_a = 2;
  int rows_b = 2;
  int cols_b = 3;

  std::vector<int> A = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> B = {1, 2, 3, 4, 5, 6};
  std::vector<int> exp_res = {9, 12, 15, 19, 26, 33, 29, 40, 51, 39, 54, 69};
  std::vector<int> C(rows_a * cols_b, 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(A.data())));
  taskDataSeq->inputs_count.emplace_back(A.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(B.data())));
  taskDataSeq->inputs_count.emplace_back(B.size());

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_a)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&cols_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&rows_b)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));
  taskDataSeq->outputs_count.emplace_back(C.size());

  shulpin_strip_scheme_A_B::Matrix_hA_vB_seq matrixTask(taskDataSeq);

  ASSERT_TRUE(matrixTask.validation());
  ASSERT_TRUE(matrixTask.pre_processing());
  ASSERT_TRUE(matrixTask.run());
  ASSERT_TRUE(matrixTask.post_processing());

  for (int i = 0; i < C.size(); ++i) {
    ASSERT_EQ(C[i], exp_res[i]);
  }
}