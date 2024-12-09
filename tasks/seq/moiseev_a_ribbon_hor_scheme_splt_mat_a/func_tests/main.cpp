#include <gtest/gtest.h>

#include "seq/moiseev_a_ribbon_hor_scheme_splt_mat_a/include/ops_seq.hpp"

template <typename DataType>
static std::vector<DataType> generateRandomValues(int size) {
  std::vector<DataType> vec(size);
  for (int i = 0; i < size; ++i) {
    vec[i] = static_cast<DataType>(rand() % 100);
  }
  return vec;
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_seq_test, test_fixed_values) {
  using DataType = int32_t;
  size_t m = 2;
  size_t k = 3;
  size_t n = 2;
  std::vector<DataType> A = {1, 2, 3, 4, 5, 6};
  std::vector<DataType> B = {7, 8, 9, 10, 11, 12};
  std::vector<DataType> C(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C.size(), m * n);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(k);
  taskData->inputs_count.emplace_back(n);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));

  moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<DataType> expected = {58, 64, 139, 154};
  EXPECT_EQ(C, expected);
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_seq_test, test_negative_fixed_values) {
  using DataType = int32_t;
  size_t m = 2;
  size_t k = 3;
  size_t n = 2;
  std::vector<DataType> A = {-3, -2, -3, -6, -9, -8};
  std::vector<DataType> B = {-7, -2, -1, -10, -3, -12};
  std::vector<DataType> C(m * n, 0);

  ASSERT_EQ(A.size(), m * k);
  ASSERT_EQ(B.size(), k * n);
  ASSERT_EQ(C.size(), m * n);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(k);
  taskData->inputs_count.emplace_back(n);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));

  moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  std::vector<DataType> expected = {32, 62, 75, 198};
  EXPECT_EQ(C, expected);
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_seq_test, test_random_values) {
  using DataType = int32_t;
  size_t m = 5;
  size_t k = 4;
  size_t n = 3;
  std::vector<DataType> A = generateRandomValues<int>(m * k);
  std::vector<DataType> B = generateRandomValues<int>(n * k);
  std::vector<DataType> C(m * n, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(k);
  taskData->inputs_count.emplace_back(n);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));

  moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_seq_test, test_float_values) {
  using DataType = float;
  size_t m = 3;
  size_t k = 3;
  size_t n = 3;
  std::vector<DataType> A = generateRandomValues<float>(m * k);
  std::vector<DataType> B = generateRandomValues<float>(n * k);
  std::vector<DataType> C(m * n, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(k);
  taskData->inputs_count.emplace_back(n);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));

  moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_seq_test, test_zero_matrix) {
  using DataType = int32_t;
  size_t m = 4;
  size_t k = 4;
  size_t n = 4;
  std::vector<DataType> A(m * k, 0);
  std::vector<DataType> B(k * n, 0);
  std::vector<DataType> C(m * n, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(k);
  taskData->inputs_count.emplace_back(n);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));

  moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());

  EXPECT_EQ(C, std::vector<DataType>(m * n, 0));
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_seq_test, test_big_mat) {
  using DataType = int32_t;
  size_t m = 10;
  size_t k = 7;
  size_t n = 8;
  std::vector<DataType> A = generateRandomValues<int>(m * k);
  std::vector<DataType> B = generateRandomValues<int>(n * k);
  std::vector<DataType> C(m * n, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(k);
  taskData->inputs_count.emplace_back(n);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));

  moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}

TEST(moiseev_a_ribbon_hor_scheme_splt_mat_a_seq_test, test_double_values) {
  using DataType = double;
  size_t m = 4;
  size_t k = 2;
  size_t n = 3;
  std::vector<DataType> A = generateRandomValues<double>(m * k);
  std::vector<DataType> B = generateRandomValues<double>(n * k);
  std::vector<DataType> C(m * n, 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(B.data()));
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(k);
  taskData->inputs_count.emplace_back(n);

  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(C.data()));

  moiseev_a_ribbon_hor_scheme_splt_mat_a_seq::MatrixMultiplicationSequential<DataType> task(taskData);

  ASSERT_TRUE(task.validation());
  ASSERT_TRUE(task.pre_processing());
  ASSERT_TRUE(task.run());
  ASSERT_TRUE(task.post_processing());
}