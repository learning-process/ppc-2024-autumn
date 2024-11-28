#include <gtest/gtest.h>

#include <random>
#include <cmath>

#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"
#include "mpi/nasedkin_e_seidels_iterate_methods/src/ops_mpi.cpp"

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

#include <random>
#include <cmath>

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_with_random_diagonally_dominant_matrix) {
  const int n = 5;  // Размер матрицы
  const double epsilon = 1e-6;

  std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
  std::vector<double> b(n, 0.0);
  std::vector<double> x(n, 0.0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A[i][j] = dist(gen);
        row_sum += std::abs(A[i][j]);
      }
    }
    A[i][i] = row_sum + dist(gen) + 1.0;
    b[i] = dist(gen);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(n);

  nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

  seidel_task.set_A(A);
  seidel_task.set_b(b);
  seidel_task.set_x(x);
  seidel_task.set_n(n);
  seidel_task.set_epsilon(epsilon);

  ASSERT_TRUE(seidel_task.validation()) << "Validation failed for random diagonally dominant matrix";
  ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed";
  ASSERT_TRUE(seidel_task.run()) << "Run failed";
  ASSERT_TRUE(seidel_task.post_processing()) << "Post-processing failed";

  const std::vector<double>& result_x = seidel_task.get_x();

  std::vector<double> Ax_minus_b(n, 0.0);
  for (int i = 0; i < n; ++i) {
    Ax_minus_b[i] = -b[i];
    for (int j = 0; j < n; ++j) {
      Ax_minus_b[i] += A[i][j] * result_x[j];
    }
  }

  double norm = 0.0;
  for (int i = 0; i < n; ++i) {
    norm += Ax_minus_b[i] * Ax_minus_b[i];
  }
  norm = std::sqrt(norm);

  ASSERT_LT(norm, epsilon) << "Solution does not satisfy ||Ax - b|| < epsilon";
}
