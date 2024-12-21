#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/nikolaev_r_strassen_matrix_multiplication_method/include/ops_seq.hpp"

TEST(nikolaev_r_strassen_matrix_multiplication_method_seq, test_2x2_matrices) {
  const size_t N = 2;

  std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> B = {5.0, 6.0, 7.0, 8.0};
  std::vector<double> expected = {19.0, 22.0, 43.0, 50.0};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs_count.emplace_back(B.size());

  std::vector<double> out(N * N, 0.0);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential>(
          taskData);

  ASSERT_TRUE(testTaskSequential->validation());
  ASSERT_TRUE(testTaskSequential->pre_processing());
  ASSERT_TRUE(testTaskSequential->run());
  ASSERT_TRUE(testTaskSequential->post_processing());

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_EQ(expected[i], out[i]);
  }
}

TEST(nikolaev_r_strassen_matrix_multiplication_method_seq, test_3x3_matrices) {
  const size_t N = 3;

  std::vector<double> A = {12.0, -25.0, 14.0, 16.0, -5.0, 4.0, -8.0, 13.0, 5.0};
  std::vector<double> B = {6.0, -5.0, 14.0, 24.0, -11.0, 4.0, 1.0, 6.0, 8.0};
  std::vector<double> expected = {-514.0, 299.0, 180.0, -20.0, -1.0, 236.0, 269.0, -73.0, -20.0};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs_count.emplace_back(B.size());

  std::vector<double> out(N * N, 0.0);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential>(
          taskData);

  ASSERT_TRUE(testTaskSequential->validation());
  ASSERT_TRUE(testTaskSequential->pre_processing());
  ASSERT_TRUE(testTaskSequential->run());
  ASSERT_TRUE(testTaskSequential->post_processing());

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_EQ(expected[i], out[i]);
  }
}

TEST(nikolaev_r_strassen_matrix_multiplication_method_seq, test_4x4_matrices) {
  const size_t N = 4;

  std::vector<double> A = {12.0, -25.0, 14.0, 4.0, 16.0, -5.0, 4.0, -6.0, -8.0, 13.0, 5.0, 11.0, -16.0, 5.0, 1.0, 2.0};
  std::vector<double> B = {6.0, -5.0, 14.0, 2.0, 24.0, -11.0, 4.0, 8.0, 1.0, 6.0, 8.0, -5.0, 12.0, 3.0, 5.0, 8.0};
  std::vector<double> expected = {-466.0, 311.0, 200.0, -214.0, -92.0, -19.0, 206.0,  -76.0,
                                  401.0,  -40.0, 35.0,  151.0,  49.0,  37.0,  -186.0, 19.0};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs_count.emplace_back(B.size());

  std::vector<double> out(N * N, 0.0);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential>(
          taskData);

  ASSERT_TRUE(testTaskSequential->validation());
  ASSERT_TRUE(testTaskSequential->pre_processing());
  ASSERT_TRUE(testTaskSequential->run());
  ASSERT_TRUE(testTaskSequential->post_processing());

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_EQ(expected[i], out[i]);
  }
}

TEST(nikolaev_r_strassen_matrix_multiplication_method_seq, test_5x5_matrices) {
  const size_t N = 5;

  std::vector<double> A = {12.0, -25.0, 14.0,  4.0, 12.0, 16.0, -5.0, 4.0,  -6.0, 5.0,  -8.0, 13.0, 5.0,
                           11.0, -4.0,  -16.0, 5.0, 1.0,  2.0,  -8.0, -1.0, 2.0,  -3.0, 4.0,  -5.0};
  std::vector<double> B = {6.0,  -5.0,  14.0, 2.0, -3.0, 24.0, -11.0, 4.0, 8.0,  5.0, 1.0,  6.0, 8.0,
                           -5.0, -15.0, 12.0, 3.0, 5.0,  8.0,  -4.0,  5.0, -4.0, 3.0, -2.0, 1.0};
  std::vector<double> expected = {-406.0, 263.0, 236.0, -238.0, -375.0, -67.0, -39.0, 221.0, -86.0,
                                  -104.0, 381.0, -24.0, 23.0,   159.0,  -34.0, 9.0,   69.0,  -210.0,
                                  35.0,   42.0,  62.0,  -3.0,   -25.0,  71.0,  37.0};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs_count.emplace_back(B.size());

  std::vector<double> out(N * N, 0.0);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential>(
          taskData);

  ASSERT_TRUE(testTaskSequential->validation());
  ASSERT_TRUE(testTaskSequential->pre_processing());
  ASSERT_TRUE(testTaskSequential->run());
  ASSERT_TRUE(testTaskSequential->post_processing());

  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_EQ(expected[i], out[i]);
  }
}

TEST(nikolaev_r_strassen_matrix_multiplication_method_seq, test_non_valid_input) {
  const size_t N = 2;

  std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> B = {5.0, 6.0, 7.0, 8.0};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs_count.emplace_back(B.size());

  std::vector<double> out(N, 0.0);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential>(
          taskData);

  ASSERT_FALSE(testTaskSequential->validation());
}

TEST(nikolaev_r_strassen_matrix_multiplication_method_seq, test_non_valid_outputs_size) {
  const size_t N = 2;

  std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> B = {5.0, 6.0, 7.0, 8.0};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs_count.emplace_back(B.size());

  std::vector<double> out(N, 0.0);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential>(
          taskData);

  ASSERT_FALSE(testTaskSequential->validation());
}

TEST(nikolaev_r_strassen_matrix_multiplication_method_seq, test_matrices_with_different_sizes) {
  std::vector<double> A = {1.0, 2.0, 3.0, 4.0};
  std::vector<double> B = {5.0, 6.0, 5.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs_count.emplace_back(B.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential>(
          taskData);

  ASSERT_FALSE(testTaskSequential->validation());
}

TEST(nikolaev_r_strassen_matrix_multiplication_method_seq, test_non_square_matrices) {
  std::vector<double> A = {1.0, 2.0, 3.0};
  std::vector<double> B = {4.0, 5.0, 6.0};

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskData->inputs_count.emplace_back(A.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskData->inputs_count.emplace_back(B.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_strassen_matrix_multiplication_method_seq::StrassenMatrixMultiplicationSequential>(
          taskData);

  ASSERT_FALSE(testTaskSequential->validation());
}