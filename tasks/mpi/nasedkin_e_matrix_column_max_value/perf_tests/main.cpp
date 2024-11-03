#include <gtest/gtest.h>
#include "ops_mpi.hpp"
#include <boost/mpi.hpp>

using namespace nasedkin_e_matrix_column_max_value_mpi;

TEST(nasedkin_e_matrix_column_max_value_mpi, test_pipeline_run) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    std::vector<std::vector<int>> matrix(1000, std::vector<int>(1000, 1));
    auto result = FindColumnMaxMPI(matrix);
    ASSERT_EQ(result.size(), 1000);
}

TEST(nasedkin_e_matrix_column_max_value_mpi, test_task_run) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    std::vector<std::vector<int>> matrix(1000, std::vector<int>(1000, 1));
    auto result = FindColumnMaxMPI(matrix);
    ASSERT_EQ(result.size(), 1000);
}
