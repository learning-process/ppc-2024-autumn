#include <mpi.h>

#include "gtest/gtest.h"
#include "mpi/konkov_i_count_words/include/ops_mpi.hpp"

namespace konkov_i_count_words_mpi {

TEST(konkov_i_count_words_mpi, test_simple) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string input = "Hello world";
  int result = countWords(input, rank);
  if (rank == 0) {
    EXPECT_EQ(result, 2);
  }
}

TEST(konkov_i_count_words_mpi, test_empty) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string input = "";
  int result = countWords(input, rank);
  if (rank == 0) {
    EXPECT_EQ(result, 0);
  }
}

TEST(konkov_i_count_words_mpi, test_single_word) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string input = "Hello";
  int result = countWords(input, rank);
  if (rank == 0) {
    EXPECT_EQ(result, 1);
  }
}

TEST(konkov_i_count_words_mpi, test_multiple_spaces) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string input = "Hello   world";
  int result = countWords(input, rank);
  if (rank == 0) {
    EXPECT_EQ(result, 2);
  }
}

TEST(konkov_i_count_words_mpi, test_leading_spaces) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string input = "   Hello world";
  int result = countWords(input, rank);
  if (rank == 0) {
    EXPECT_EQ(result, 2);
  }
}

TEST(konkov_i_count_words_mpi, test_trailing_spaces) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string input = "Hello world   ";
  int result = countWords(input, rank);
  if (rank == 0) {
    EXPECT_EQ(result, 2);
  }
}

}  // namespace konkov_i_count_words_mpi