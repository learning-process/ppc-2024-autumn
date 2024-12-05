#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>

#include "core/task/include/task.hpp"

namespace kazunin_n_dining_philosophers_mpi {

enum State { THINKING, HUNGRY, EATING };

enum MessageTag { REQUEST_FORK, FORK_AVAILABLE };

class DiningPhilosophersParallelMPI : public ppc::core::Task {
 public:
  explicit DiningPhilosophersParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
};

}  // namespace kazunin_n_dining_philosophers_mpi
