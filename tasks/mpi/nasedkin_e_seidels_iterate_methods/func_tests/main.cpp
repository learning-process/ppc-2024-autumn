#include <gtest/gtest.h>

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

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_random_diagonal_dominant_matrix) {
  const int n = 10;
  const double epsilon = 1e-6;

  std::vector<std::vector<double>> A(n, std::vector<double>(n));
  std::vector<double> b(n);

  std::srand(static_cast<unsigned>(std::time(nullptr)));
  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        A[i][j] = static_cast<double>(std::rand() % 10 + 1);
        sum += std::abs(A[i][j]);
      }
    }
    A[i][i] = sum + static_cast<double>(std::rand() % 10 + 1);
    b[i] = static_cast<double>(std::rand() % 20 + 1);
  }

  for (int i = 0; i < n; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        row_sum += std::abs(A[i][j]);
      }
    }
    ASSERT_GT(std::abs(A[i][i]), row_sum) << "Matrix is not diagonally dominant";
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();

  nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

  seidel_task.set_matrix(A);
  seidel_task.set_vector(b);

  ASSERT_TRUE(seidel_task.validation()) << "Validation failed for valid random diagonal dominant matrix";
  ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed";
  ASSERT_TRUE(seidel_task.run()) << "Run failed";
  ASSERT_TRUE(seidel_task.post_processing()) << "Post-processing failed";

  std::vector<double> x = seidel_task.get_solution();

  std::vector<double> Ax(n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      Ax[i] += A[i][j] * x[j];
    }
  }

  double norm = 0.0;
  for (int i = 0; i < n; ++i) {
    norm += (Ax[i] - b[i]) * (Ax[i] - b[i]);
  }
  norm = std::sqrt(norm);

  ASSERT_LT(norm, epsilon) << "The solution does not satisfy ||Ax - b|| < eps";
}

