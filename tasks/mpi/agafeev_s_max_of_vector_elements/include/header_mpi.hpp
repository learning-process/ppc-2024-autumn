#pragma once
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace agafeev_s_max_of_vector_elements_mpi {

std::vector<int> create_RandomMatrix(int row_size, int column_size);

int get_MaxValue(std::vector<int> matrix);

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
};

class MaxMatrixMpi : public ppc::core::Task {
 public:
  explicit MaxMatrixMpi(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  std::vector<int> input_;
  int maxres_{};
};

}  // namespace agafeev_s_max_of_vector_elements_mpi