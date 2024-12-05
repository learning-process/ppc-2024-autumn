#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"

TEST(kazunin_n_dining_philosophers_mpi, 5_second_simulation_test) {
  boost::mpi::communicator world;
  if (world.size() >= 3) {
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    auto taskParallel = std::make_shared<kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI>(taskDataPar);
    taskParallel->validation();
    taskParallel->pre_processing();
    EXPECT_TRUE(taskParallel->run());
    taskParallel->post_processing();
  } else {
    EXPECT_TRUE(true);
  }
}
