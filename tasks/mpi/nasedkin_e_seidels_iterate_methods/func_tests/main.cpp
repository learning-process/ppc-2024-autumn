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

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_random_diag_dominant_matrix) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(5);

  nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

  std::vector<std::vector<double>> matrix;
  std::vector<double> vector;
  nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI::generate_random_diag_dominant_matrix(5, matrix, vector);
  seidel_task.set_matrix(matrix, vector);

  ASSERT_TRUE(seidel_task.validation()) << "Validation failed for random diagonally dominant matrix";

  ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed for random diagonally dominant matrix";

  ASSERT_TRUE(seidel_task.run()) << "Run failed for random diagonally dominant matrix";

  ASSERT_TRUE(seidel_task.post_processing()) << "Post-processing failed for random diagonally dominant matrix";
}
