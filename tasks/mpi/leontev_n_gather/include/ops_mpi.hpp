// Copyright 2023 Nesterov Alexander
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>

#include "core/task/include/task.hpp"

namespace leontev_n_mat_vec_mpi {

class MPIMatVecSequential : public ppc::core::Task {
 public:
  explicit MPIMatVecSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> vec_;
  std::vector<int> mat_;
  std::vector<int> res;
  std::string ops;
};

class MPIMatVecParallel : public ppc::core::Task {
 public:
  explicit MPIMatVecParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;
  template <class InOutType>
  void my_gather(const boost::mpi::communicator& wrld, const std::vector<InOutType>& input, InOutType* output,
                 const std::vector<int>& sizes, int root);

 private:
  std::vector<int> vec_;
  std::vector<int> mat_;
  std::vector<int> res;
  std::vector<int> local_tmp_;
  std::string ops;
  boost::mpi::communicator world;
};

}  // namespace leontev_n_mat_vec_mpi
