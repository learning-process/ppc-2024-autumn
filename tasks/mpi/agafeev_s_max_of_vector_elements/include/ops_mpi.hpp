#pragma once
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
// #include "seq/agafeev_s_max_of_vector_elements/include/ops_seq.hpp"

namespace agafeev_s_max_of_vector_elements_mpi {

template <typename T>
std::vector<T> create_RandomMatrix(int row_size, int column_size) {
  auto rand_gen = std::mt19937(1337);
  std::vector<T> matrix(row_size * column_size);
  for (unsigned int i = 0; i < matrix.size(); i++) matrix[i] = rand_gen() % 100;

  return matrix;
}

template <typename T>
T get_MaxValue(std::vector<T> matrix);

/*
class MaxMatrixSeq : public ppc::core::Task {
 public:
  explicit MaxMatrixSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int maxres_{};
};*/

template <typename T>
class MaxMatrixMpi : public ppc::core::Task {
 public:
  explicit MaxMatrixMpi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<T> input_;
  std::vector<T> local_vector;
  T maxres_{};
};

}  // namespace agafeev_s_max_of_vector_elements_mpi