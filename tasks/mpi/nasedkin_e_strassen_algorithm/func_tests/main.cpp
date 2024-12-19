#include <gtest/gtest.h>

#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"
#include "mpi/nasedkin_e_strassen_algorithm/src/ops_mpi.cpp"

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_2x2) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(2);

    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(2, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(2, matrixB);
    strassen_task.set_matrices(matrixA, matrixB);

    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_4x4) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(4);

    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(4, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(4, matrixB);
    strassen_task.set_matrices(matrixA, matrixB);

    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}

TEST(nasedkin_e_strassen_algorithm_mpi, test_random_matrix_8x8) {
    auto taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs_count.push_back(8);

    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI strassen_task(taskData);

    std::vector<std::vector<double>> matrixA;
    std::vector<std::vector<double>> matrixB;
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(8, matrixA);
    nasedkin_e_strassen_algorithm::StrassenAlgorithmMPI::generate_random_matrix(8, matrixB);
    strassen_task.set_matrices(matrixA, matrixB);

    ASSERT_TRUE(strassen_task.validation()) << "Validation failed for random matrix";
    ASSERT_TRUE(strassen_task.pre_processing()) << "Pre-processing failed for random matrix";
    ASSERT_TRUE(strassen_task.run()) << "Run failed for random matrix";
    ASSERT_TRUE(strassen_task.post_processing()) << "Post-processing failed for random matrix";
}