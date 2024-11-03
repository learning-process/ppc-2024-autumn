#include <mpi.h>

#include "gtest/gtest.h"
#include "mpi/konkov_i_count_words/include/ops_mpi.hpp"

namespace konkov_i_count_words_mpi {

TEST(konkov_i_count_words_mpi, test_performance) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::string input = "Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
  for (int i = 0; i < 1000000; ++i) {
    countWords(input, rank);
  }
}

}  // namespace konkov_i_count_words_mpi