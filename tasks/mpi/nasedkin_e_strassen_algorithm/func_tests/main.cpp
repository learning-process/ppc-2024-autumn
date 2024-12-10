#include <gtest/gtest.h>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <vector>
#include <random>
#include "mpi/nasedkin_e_strassen_algorithm/include/ops_mpi.hpp"

std::vector<double> generate_dense_matrix(size_t n, double min_val = 1.0, double max_val = 10.0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(min_val, max_val);
    std::vector<double> matrix(n * n);
    for (size_t i = 0; i < n * n; ++i) {
        matrix[i] = dis(gen);
    }
    return matrix;
}

std::vector<double> generate_diagonal_zero_matrix(size_t n, double min_val = 1.0, double max_val = 10.0) {
    auto matrix = generate_dense_matrix(n, min_val, max_val);
    for (size_t i = 0; i < n; ++i) {
        matrix[i * n + i] = 0.0;
    }
    return matrix;
}

void test_matrix_multiplication(const std::vector<double>& matrix_a, const std::vector<double>& matrix_b, size_t n) {
    boost::mpi::communicator world;

    auto taskData = std::make_shared<ppc::core::TaskData>();
    std::vector<double> result_matrix(n * n, 0);

    if (world.rank() == 0) {
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(matrix_a.data())));
        taskData->inputs_count.push_back(n);
        taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(matrix_b.data())));
        taskData->inputs_count.push_back(n);
        taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_matrix.data()));
        taskData->outputs_count.push_back(n);
    }

    nasedkin_e_strassen_algorithm::TestMPITaskParallel task(taskData);
    ASSERT_TRUE(task.validation());
    task.pre_processing();
    task.run();
    task.post_processing();

    if (world.rank() == 0) {
        for (size_t i = 0; i < n * n; ++i) {
            ASSERT_TRUE(std::isfinite(result_matrix[i]));
        }
    }
}

TEST(MPI_Strassen_Test, Test_2x2) {
    size_t n = 2;
    auto matrix_a = generate_dense_matrix(n);
    auto matrix_b = generate_dense_matrix(n);
    test_matrix_multiplication(matrix_a, matrix_b, n);
}

TEST(MPI_Strassen_Test, Test_4x4) {
    size_t n = 4;
    auto matrix_a = generate_dense_matrix(n);
    auto matrix_b = generate_dense_matrix(n);
    test_matrix_multiplication(matrix_a, matrix_b, n);
}

TEST(MPI_Strassen_Test, Test_8x8) {
    size_t n = 8;
    auto matrix_a = generate_dense_matrix(n);
    auto matrix_b = generate_dense_matrix(n);
    test_matrix_multiplication(matrix_a, matrix_b, n);
}

TEST(MPI_Strassen_Test, Test_4x4_Diagonal_Zero) {
    size_t n = 4;
    auto matrix_a = generate_diagonal_zero_matrix(n);
    auto matrix_b = generate_dense_matrix(n);
    test_matrix_multiplication(matrix_a, matrix_b, n);
}
