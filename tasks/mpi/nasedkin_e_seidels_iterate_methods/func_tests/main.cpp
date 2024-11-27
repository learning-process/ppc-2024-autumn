#include <gtest/gtest.h>

#include <random>
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

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_random_matrix_solution_accuracy) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(5);

  nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

  ASSERT_TRUE(seidel_task.validation()) << "Validation failed for valid input";
  ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed";

  std::vector<std::vector<double>> A(taskData->inputs_count[0], std::vector<double>(taskData->inputs_count[0]));
  std::vector<double> b(taskData->inputs_count[0]);
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  for (int i = 0; i < taskData->inputs_count[0]; ++i) {
    for (int j = 0; j < taskData->inputs_count[0]; ++j) {
      A[i][j] = (i == j) ? dist(rng) + 15.0 : dist(rng);
    }
    b[i] = dist(rng);
  }

  seidel_task.set_matrix(A);
  seidel_task.set_vector(b);

  ASSERT_TRUE(seidel_task.run()) << "Run failed";

  const auto& x = seidel_task.get_solution();
  double norm = 0.0;

  for (int i = 0; i < taskData->inputs_count[0]; ++i) {
    double Ax_i = 0.0;
    for (int j = 0; j < taskData->inputs_count[0]; ++j) {
      Ax_i += A[i][j] * x[j];
    }
    norm += (Ax_i - b[i]) * (Ax_i - b[i]);
  }

  norm = std::sqrt(norm);
  ASSERT_LT(norm, 1e-6) << "Solution accuracy is not within the expected threshold";
}