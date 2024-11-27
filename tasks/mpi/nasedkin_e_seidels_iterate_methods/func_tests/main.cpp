#include <gtest/gtest.h>


#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"
#include "mpi/nasedkin_e_seidels_iterate_methods/src/ops_mpi.cpp"

#include <random>

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_with_valid_input) {
auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(3);

  nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

  ASSERT_TRUE(seidel_task.validation()) << "Validation failed for valid input";
  ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed";
  ASSERT_TRUE(seidel_task.run()) << "Run failed";
  ASSERT_TRUE(seidel_task.post_processing()) << "Post-processing failed";
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_with_invalid_input) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(0);

  nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

  ASSERT_FALSE(seidel_task.validation());
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_with_negative_input) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(static_cast<unsigned>(-3));

  nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

  ASSERT_FALSE(seidel_task.validation()) << "Validation passed for negative input, expected failure";
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_matrix_with_zero_diagonal) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(3);
  taskData->inputs_count.push_back(0);

  nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

  ASSERT_TRUE(seidel_task.validation()) << "Validation failed for valid input";
  ASSERT_FALSE(seidel_task.pre_processing()) << "Pre-processing passed, but expected failure";
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_correct_matrix_random) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(10);

  nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

  int n = 10;
  std::vector<std::vector<double>> A(n, std::vector<double>(n));
  std::vector<double> b(n);

  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(-10, 10);

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      A[i][j] = (i == j) ? 0.0 : dis(gen);
      row_sum += std::abs(A[i][j]);
    }
    A[i][i] = row_sum + dis(gen);
    b[i] = dis(gen);
  }

  seidel_task.set_matrix(A, b);

  ASSERT_TRUE(seidel_task.validation()) << "Validation failed for valid input";
  ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed";
  ASSERT_TRUE(seidel_task.run()) << "Run failed";

  double residual_norm = seidel_task.calculate_residual_norm();
  ASSERT_LT(residual_norm, 1e-6) << "Residual norm ||Ax - b|| not within epsilon";
}

