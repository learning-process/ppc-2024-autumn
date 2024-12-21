#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <climits>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm {
/*
void multiplyBlock(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int blockSize,
                   int n, int p) {
  for (int i = 0; i < blockSize; ++i) {
    for (int j = 0; j < p; ++j) {
      C[i * p + j] = 0.0;
      for (int k = 0; k < n; ++k) {
        C[i * p + j] += A[i * n + k] * B[k * p + j];
      }
    }
  }
}
*/
class dense_matrix_multiplication_block_scheme_fox_algorithm_seq : public ppc::core::Task {
 public:
  explicit dense_matrix_multiplication_block_scheme_fox_algorithm_seq(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A;
  int N;
  std::vector<double> B;
  std::vector<double> C;
};

class dense_matrix_multiplication_block_scheme_fox_algorithm_mpi : public ppc::core::Task {
 public:
  explicit dense_matrix_multiplication_block_scheme_fox_algorithm_mpi(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<double> A;
  std::vector<double> B;
  int N;
  std::vector<double> C;
  boost::mpi::communicator world;
};

}  // namespace alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm