#include <gtest/gtest.h>

#include "../include/matrix_operations.hpp"
#include "../include/tests.hpp"

using namespace khasanyanov_k_fox_algorithm;

TEST(khasanyanov_k_matrix_tests, test_wrong_sizes) {
  matrix<int> A{3, 2, {5, 6, 7, 8, 5, 6}};
  matrix<int> B{4, 3, {8, 7, 8, 5, 87, 7, 6, 9, 0, 9, 79, 7}};
  matrix<int> C{2, 5};
  ASSERT_ANY_THROW(MatrixOperations::multiply(A, B));
  ASSERT_NO_THROW(MatrixOperations::multiply(A, C));
}

TEST(khasanyanov_k_matrix_tests, test_cant_add) {
  matrix<int> A{3, 2, {5, 6, 7, 8, 5, 6}};
  matrix<int> B{4, 3, {8, 7, 8, 5, 87, 7, 6, 9, 0, 9, 79, 7}};
  EXPECT_FALSE(A == B);
  ASSERT_ANY_THROW(A += B);
}

TEST(khasanyanov_k_matrix_tests, test_int_3_4_4_3) {
  matrix<int> A{3, 4, {5, 6, 7, 8, 5, 6, 8, 89, 8, 4, 6, 6}};
  matrix<int> B{4, 3, {8, 7, 8, 5, 87, 7, 6, 9, 0, 9, 79, 7}};
  matrix<int> expected_solution{3, 3, {184, 1252, 138, 919, 7660, 705, 174, 932, 134}};
  matrix<int> actual_solution = MatrixOperations::multiply(A, B);
  ASSERT_EQ(expected_solution, actual_solution);
}

TEST(khasanyanov_k_matrix_tests, test_int_2_2_2_2) {
  matrix<int> A{2, 2, {1, 2, -3, 4}};
  matrix<int> B{2, 2, {-2, 4, 3, 1}};
  matrix<int> expected_solution{2, 2, {4, 6, 18, -8}};
  matrix<int> actual_solution = MatrixOperations::multiply(A, B);
  ASSERT_EQ(expected_solution, actual_solution);
}

TEST(khasanyanov_k_matrix_tests, test_double_2_3_3_1) {
  matrix<double> A{2, 3, {1.1, 2.2, 3.3, 4.4, 5.5, 6.6}};
  matrix<double> B{3, 1, {11.12, 12.13, 13.14}};
  matrix<double> expected_solution{2, 1, {82.28, 202.367}};
  matrix<double> actual_solution = MatrixOperations::multiply(A, B);
  for (size_t i = 0; i < expected_solution.size(); ++i) {
    EXPECT_NEAR(expected_solution[i], actual_solution[i], 1e05);
  }
}

TEST(khasanyanov_k_matrix_tests, test_hard_double_4_5_5_6) {
  matrix<double> A{4, 5, {543, 67.89, 54, 98.09, 543.2, 7.6, 4, 89, 3, 123.4, 21, 65, 8, 543.6, 35.7, 7, 3, 7, 8, 9}};
  matrix<double> B{
      5, 6, {11.12, 43.76, 86.7, 43,    12,      54,   12.13, 65.87, 876.2, 65.8,  0,  54,  13.14, 43.2, 654.86,
             75.77, 87.6,  43,   543.2, 321.422, 76.4, 0.64,  0.678, 8,     321.5, 26, 785, 65,    3,    7}};
  matrix<double> expected_solution{
      4, 6, {235492.5137, 76217.87828, 575831.834, 67278.5196, 12942.50502, 39897.18,    42605.192, 8613.522,
             159544.46,   15356.45,    8259.834,   5341.2,     307888.16,   181199.3092, 133568.12, 8454.564,
             1428.4608,   9586.7,      7445.31,    3611.706,   15495.72,    1618.91,     729.624,   968}};
  matrix<double> actual_solution = MatrixOperations::multiply(A, B);
  for (size_t i = 0; i < expected_solution.size(); ++i) {
    EXPECT_NEAR(expected_solution[i], actual_solution[i], 1e05);
  }
}

TEST(khasanyanov_k_matrix_tests, test_int_sequential_multiply) {
  matrix<int> A{3, 4, {5, 6, 7, 8, 5, 6, 8, 89, 8, 4, 6, 6}};
  matrix<int> B{4, 3, {8, 7, 8, 5, 87, 7, 6, 9, 0, 9, 79, 7}};
  matrix<int> expected_solution{3, 3, {184, 1252, 138, 919, 7660, 705, 174, 932, 134}};
  matrix<int> actual_solution(3, 3);
  std::shared_ptr<ppc::core::TaskData> taskData = create_task_data(A, B, actual_solution);
  MatrixMultiplication<int> test(taskData);
  RUN_TASK(test);
  ASSERT_EQ(expected_solution, actual_solution);
}