#pragma once
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <ctime>
#include <limits>
#include <random>
#include <string>
#include <vector>
// Will delete
#include <iostream>

#include "core/task/include/task.hpp"

namespace agafeev_s_max_of_vector_elements_mpi {

template <typename T>
T get_MaxValue(std::vector<T> matrix) {
  T max_result = std::numeric_limits<T>::min();
  for (unsigned int i = 0; i < matrix.size(); ++i)
    if (max_result < matrix[i]) max_result = matrix[i];

  return max_result;
}

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
  int lv_size{};
};

template <typename T>
class MaxMatrixSeq : public ppc::core::Task {
 public:
  explicit MaxMatrixSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<T> input_;
  T maxres_{};
};

}  // namespace agafeev_s_max_of_vector_elements_mpi