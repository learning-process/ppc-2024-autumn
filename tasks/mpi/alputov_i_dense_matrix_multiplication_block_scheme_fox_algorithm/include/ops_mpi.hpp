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
  std::vector<double> rectA;
  std::vector<double> rectB;
  std::vector<double> localA;
  std::vector<double> localB;
  std::vector<double> mult_block;
  std::vector<double> block_A_to_send;
  std::vector<double> block_B_to_send;
  std::vector<double> resultM;
  int N;
  std::vector<double> C;
  boost::mpi::communicator world;
};

}  // namespace alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm