#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace burykin_m_broadcast_mpi_mpi {

class StdBroadcastMPI : public ppc::core::Task {
 public:
  explicit StdBroadcastMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_vector;
  int global_max;
  int source_worker;
  boost::mpi::communicator world;
};

}  // namespace burykin_m_broadcast_mpi_mpi
