#include <gtest/gtest.h>
#include "ops_mpi.hpp"
#include <boost/mpi.hpp>

using namespace nasedkin_e_matrix_column_max_value_mpi;

TEST(nasedkin_e_matrix_column_max_value_mpi, find_max_in_each_column) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    std::vector<std::vector<int>> matrix = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
    };
    std::vector<int> expected = {7, 8, 9};
    EXPECT_EQ(FindColumnMaxMPI(matrix), expected);
}

TEST(nasedkin_e_matrix_column_max_value_mpi, handle_empty_matrix) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    std::vector<std::vector<int>> matrix = {};
    std::vector<int> expected = {};
    EXPECT_EQ(FindColumnMaxMPI(matrix), expected);
}

TEST(nasedkin_e_matrix_column_max_value_mpi, handle_single_column) {
    boost::mpi::environment env;
    boost::mpi::communicator world;

    std::vector<std::vector<int>> matrix = {
        {3},
        {1},
        {4},
    };
    std::vector<int> expected = {4};
    EXPECT_EQ(FindColumnMaxMPI(matrix), expected);
}
