#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>

#include "core/task/include/task.hpp"

namespace kazunin_n_dining_philosophers_mpi {

enum State : std::uint8_t { THINKING, HUNGRY, EATING };

enum MessageTag : std::uint8_t { REQUEST_FORK, FORK_AVAILABLE };

class DiningPhilosophersParallelMPI : public ppc::core::Task {
 public:
  explicit DiningPhilosophersParallelMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int rank;
  int size;
  int leftNeighbor;
  int rightNeighbor;
  bool hasLeftFork;
  bool hasRightFork;
  bool leftForkAvailable;
  bool rightForkAvailable;
  double SIMULATION_TIME;
  int SLEEP_TIME_MS;
  State state;
  MPI_Status status;
  int thinkTime;
  int eatTime;
  int timeCounter;
  boost::mpi::communicator world;
};

}  // namespace kazunin_n_dining_philosophers_mpi
