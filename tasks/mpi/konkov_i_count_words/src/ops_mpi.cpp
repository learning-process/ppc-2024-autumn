#include "mpi/konkov_i_count_words/include/ops_mpi.hpp"

#include <mpi.h>

#include <sstream>

namespace konkov_i_count_words_mpi {

int countWords(const std::string& input, int rank) {
  std::istringstream stream(input);
  std::string word;
  int count = 0;
  while (stream >> word) {
    ++count;
  }
  int global_count;
  MPI_Reduce(&count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  return global_count;
}

}  // namespace konkov_i_count_words_mpi