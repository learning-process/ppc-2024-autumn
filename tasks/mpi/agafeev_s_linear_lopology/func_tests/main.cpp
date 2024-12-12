#include <gtest/gtest.h>

#include "mpi/agafeev_s_linear_lopology/include/lintop_mpi.hpp"

template <typename T>
static std::vector<T> create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(std::time(nullptr));
  std::vector<T> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); ++i) matrix[i] = rand_gen() % 200 - 100;

  return matrix;
}