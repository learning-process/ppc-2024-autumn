#ifndef OPS_MPI_HPP
#define OPS_MPI_HPP

#include <string>

namespace konkov_i_count_words_mpi {

int countWords(const std::string& input, int rank);

}  // namespace konkov_i_count_words_mpi

#endif  // OPS_MPI_HPP