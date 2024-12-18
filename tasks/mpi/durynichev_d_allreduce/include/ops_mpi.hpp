#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace durynichev_d_allreduce_mpi {

class MpiAllreduceMPI : public ppc::core::Task {
 public:
  explicit MpiAllreduceMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> data;
  int global_sum = 0;
  boost::mpi::communicator world;
};

}  // namespace durynichev_d_allreduce_mpi
