#include <gtest/gtest.h>
#include "ops_mpi.hpp"
#include <mpi.h>
#include <chrono>
#include <iostream>

namespace nasedkin_e_matrix_column_max_value_mpi {

    TEST(nasedkin_e_matrix_column_max_value_mpi, test_task_run) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::vector<int> matrix(1000 * 1000, 1);
        int rows = 1000, cols = 1000;

        auto start = std::chrono::high_resolution_clock::now();
        find_max_by_columns(matrix, rows, cols, rank, 1);
        auto end = std::chrono::high_resolution_clock::now();

        if (rank == 0) {
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "RunTask time: " << elapsed.count() << " seconds\n";
        }
    }

    TEST(nasedkin_e_matrix_column_max_value_mpi, test_pipeline_run) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        std::vector<int> matrix(1000 * 1000, 1);
        int rows = 1000, cols = 1000;

        auto start = std::chrono::high_resolution_clock::now();
        find_max_by_columns(matrix, rows, cols, rank, 1);
        auto end = std::chrono::high_resolution_clock::now();

        if (rank == 0) {
            std::chrono::duration<double> elapsed = end - start;
            std::cout << "Pipeline time: " << elapsed.count() << " seconds\n";
        }
    }

}
