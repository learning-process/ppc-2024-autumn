#include <gtest/gtest.h>

#include "mpi/nasedkin_e_seidels_iterate_methods/include/ops_mpi.hpp"
#include "mpi/nasedkin_e_seidels_iterate_methods/src/ops_mpi.cpp"

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_matrix_with_zero_diagonal) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(3);
    taskData->inputs_count.push_back(0);

    ASSERT_EQ(static_cast<int>(taskData->inputs_count.size()), 2) << "Inputs count size should be exactly 2";
    ASSERT_GT(static_cast<int>(taskData->inputs_count[0]), 0) << "Matrix size should be greater than 0";

    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

    ASSERT_TRUE(seidel_task.validation()) << "Validation failed for valid input";
    ASSERT_FALSE(seidel_task.pre_processing()) << "Pre-processing passed, but expected failure";
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_random_matrix_2x2) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(2);

    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

    std::vector<std::vector<double>> matrix;
    std::vector<double> vector;
    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI::generate_random_matrix(2, matrix, vector);
    seidel_task.set_matrix(matrix, vector);

    ASSERT_TRUE(seidel_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(seidel_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(seidel_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_random_matrix_3x3) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(3);

    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

    std::vector<std::vector<double>> matrix;
    std::vector<double> vector;
    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI::generate_random_matrix(3, matrix, vector);
    seidel_task.set_matrix(matrix, vector);

    ASSERT_TRUE(seidel_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(seidel_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(seidel_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_random_matrix_5x5) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(5);

    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

    std::vector<std::vector<double>> matrix;
    std::vector<double> vector;
    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI::generate_random_matrix(5, matrix, vector);
    seidel_task.set_matrix(matrix, vector);

    ASSERT_TRUE(seidel_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(seidel_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(seidel_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_seidels_iterate_methods_mpi, test_random_matrix_10x10) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(10);

    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI seidel_task(taskData);

    std::vector<std::vector<double>> matrix;
    std::vector<double> vector;
    nasedkin_e_seidels_iterate_methods_mpi::SeidelIterateMethodsMPI::generate_random_matrix(10, matrix, vector);
    seidel_task.set_matrix(matrix, vector);

    ASSERT_TRUE(seidel_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(seidel_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(seidel_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(seidel_task.post_processing()) << "Post-processing failed for random matrix";
}