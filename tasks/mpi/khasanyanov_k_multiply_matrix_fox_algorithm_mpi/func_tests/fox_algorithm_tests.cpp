#include <gtest/gtest.h>

#include <memory>

#include "../include/fox_algorithm.hpp"
#include "../include/matrix_operations.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"
#include "mpi/khasanyanov_k_multiply_matrix_fox_algorithm_mpi/include/matrix.hpp"
#include "mpi/khasanyanov_k_multiply_matrix_fox_algorithm_mpi/include/tests.hpp"

using namespace khasanyanov_k_fox_algorithm;

TEST(khasanyanov_k_fox_algorithm_tests, test_calculate_processors_grid) {
  auto grid1 = FoxAlgorithm<>::get_processors_grid(11);
  auto expect_grid1 = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8};
  auto grid2 = FoxAlgorithm<>::get_processors_grid(4);
  auto expect_grid2 = std::vector<int>{0, 1, 2, 3};
  auto grid3 = FoxAlgorithm<>::get_processors_grid(2);
  auto expect_grid3 = std::vector<int>{0};
  EXPECT_EQ(grid1, expect_grid1);
  EXPECT_EQ(grid2, expect_grid2);
  EXPECT_EQ(grid3, expect_grid3);
}

TEST(khasanyanov_k_fox_algorithm_tests, test_calculate_block_size) {
  matrix<int> mt1{3, 4};
  matrix<int> mt2{4, 3};
  matrix<int> mt3{10, 5};
  matrix<int> mt4{5, 1};
  matrix<int> mt5{2, 6};
  matrix<int> mt6{6, 9};
  auto size1 = FoxAlgorithm<>::calculate_block_size(4, mt1, mt2);
  auto size2 = FoxAlgorithm<>::calculate_block_size(4, mt3, mt4);
  auto size3 = FoxAlgorithm<>::calculate_block_size(4, mt5, mt6);
  auto size4 = FoxAlgorithm<>::calculate_block_size(9, mt1, mt2);
  auto size5 = FoxAlgorithm<>::calculate_block_size(9, mt3, mt4);
  auto size6 = FoxAlgorithm<>::calculate_block_size(1, mt5, mt6);
  EXPECT_EQ(2ull, size1);
  EXPECT_EQ(5ull, size2);
  EXPECT_EQ(5ull, size3);
  EXPECT_EQ(2ull, size4);
  EXPECT_EQ(4ull, size5);
  EXPECT_EQ(9ull, size6);
}

TEST(khasanyanov_k_fox_algorithm_tests, test_distribution) {
  std::vector<int> vec(20);
  std::iota(vec.begin(), vec.end(), 1);
  matrix<int> mt{4, 5, vec};
  auto res = FoxAlgorithm<int>::get_block_grid(2, 9, mt);
  BlockGrid<int> expected = {{2, 2, {1, 2, 6, 7}},     {2, 2, {3, 4, 8, 9}},     {2, 2, {5, 0, 10, 0}},
                             {2, 2, {11, 12, 16, 17}}, {2, 2, {13, 14, 18, 19}}, {2, 2, {15, 0, 20, 0}},
                             {2, 2, {0, 0, 0, 0}},     {2, 2, {0, 0, 0, 0}},     {2, 2, {0, 0, 0, 0}}};
  EXPECT_EQ(expected, res);
}

TEST(khasanyanov_k_fox_algorithm_tests, test_convertion_grid_to_matrix) {
  boost::mpi::communicator world;
  matrix<int> M00 = {2, 2, {1, 2, 4, 5}};
  matrix<int> M01 = {2, 2, {3, 0, 6, 0}};
  matrix<int> M10 = {2, 2, {7, 8, 0, 0}};
  matrix<int> M11 = {2, 2, {9, 0, 0, 0}};
  BlockGrid<int> grid = {M00, M01, M10, M11};
  matrix<int> c(3, 3);
  FoxAlgorithm<int>::convert_to_matrix(grid, 2, c);
  matrix<int> expected{3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9}};
  EXPECT_EQ(expected, c);
}

TEST(khasanyanov_k_fox_algorithm_tests, test_int) {
  boost::mpi::communicator world;
  matrix<int> A{3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  matrix<int> B{4, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  matrix<int> C{3, 3};
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = create_task_data(A, B, C);
  }
  FoxAlgorithm<int> test(taskData);
  RUN_TASK(test);
  if (world.rank() == 0) {
    auto expected = MatrixOperations::multiply(A, B);
    EXPECT_EQ(expected, C);
  }
}

TEST(khasanyanov_k_fox_algorithm_tests, test_validation) {
  boost::mpi::communicator world;
  matrix<int> A{3, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  matrix<int> B{4, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  // can't mult these matrix
  matrix<int> C{0, 2};  // invalid output size
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = create_task_data(A, B, C);
  }
  FoxAlgorithm<int> test(taskData);
  if (world.rank() == 0) {
    EXPECT_FALSE(test.validation());
  }
}

TEST(khasanyanov_k_fox_algorithm_tests, test_mult_zero_matrix) {
  boost::mpi::communicator world;
  matrix<int> A{3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
  matrix<int> B{4, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  matrix<int> C{3, 3};
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = create_task_data(A, B, C);
  }
  FoxAlgorithm<int> test(taskData);
  RUN_TASK(test);
  if (world.rank() == 0) {
    auto expected = matrix<int>{3, 3, {0, 0, 0, 0, 0, 0, 0, 0, 0}};
    ASSERT_EQ(expected, C);
  }
}

class khasanyanov_k_fox_algorithm_tests : public ::testing::TestWithParam<std::tuple<size_t, size_t, size_t, size_t>> {
 protected:
  boost::mpi::communicator world;
  long double abs_error = 1e05;
  matrix<double> A;
  matrix<double> B;
};

TEST_P(khasanyanov_k_fox_algorithm_tests, test_multiply) {
  const size_t m1 = std::get<0>(GetParam());
  const size_t n1 = std::get<1>(GetParam());
  const size_t m2 = std::get<2>(GetParam());
  const size_t n2 = std::get<3>(GetParam());
  A = MatrixOperations::generate_random_matrix<double>(m1, n1, -1000, 1000);
  B = MatrixOperations::generate_random_matrix<double>(m2, n2, -1000, 1000);
  matrix<double> C{m1, n2};
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData = create_task_data(A, B, C);
  }
  FoxAlgorithm<double> test(taskData);
  RUN_TASK(test);
  if (world.rank() == 0) {
    auto expected = MatrixOperations::multiply(A, B);
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], C[i], abs_error);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(khasanyanov_k_fox_algorithm, khasanyanov_k_fox_algorithm_tests,
                         ::testing::Values(std::make_tuple(3, 3, 3, 3), std::make_tuple(3, 4, 4, 3),
                                           std::make_tuple(10, 10, 10, 10), std::make_tuple(15, 32, 32, 32),
                                           std::make_tuple(1, 1, 1, 1), std::make_tuple(100, 100, 100, 100),
                                           std::make_tuple(64, 23, 23, 45), std::make_tuple(1, 10, 10, 2),
                                           std::make_tuple(100, 49, 49, 12), std::make_tuple(100, 200, 200, 100),
                                           std::make_tuple(76, 56, 56, 34), std::make_tuple(256, 256, 256, 256),
                                           std::make_tuple(64, 64, 64, 64), std::make_tuple(1, 256, 256, 2)));
