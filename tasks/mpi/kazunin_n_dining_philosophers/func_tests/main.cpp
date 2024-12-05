#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"

namespace kazunin_n_dining_philosophers_mpi {
void run_simulation(double simulation_time = 0.6, double sleep_time = 0.001) {
  boost::mpi::communicator world;
  if (world.size() >= 3) {
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&simulation_time));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sleep_time));
    taskDataPar->inputs_count.emplace_back(1);

    auto taskParallel = std::make_shared<kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI>(taskDataPar);
    if (simulation_time < 1 && sleep_time < simulation_time) {
      EXPECT_TRUE(taskParallel->validation());
      taskParallel->pre_processing();
      taskParallel->run();
      taskParallel->post_processing();
    } else {
      EXPECT_FALSE(taskParallel->validation());
    }
  } else {
    EXPECT_TRUE(true);
  }
}
}  // namespace kazunin_n_dining_philosophers_mpi

TEST(kazunin_n_dining_philosophers_mpi, 900_millisecond_simulation_test) {
  kazunin_n_dining_philosophers_mpi::run_simulation(0.9);
}

TEST(kazunin_n_dining_philosophers_mpi, 900_millisecond_simulation_and_10_millisecond_sleep_test) {
  kazunin_n_dining_philosophers_mpi::run_simulation(0.9, 0.01);
}

TEST(kazunin_n_dining_philosophers_mpi, 100_millisecond_simulation_and_1_millisecond_sleep_test) {
  kazunin_n_dining_philosophers_mpi::run_simulation(0.1, 0.001);
}

TEST(kazunin_n_dining_philosophers_mpi, 10_millisecond_simulation_and_100_millisecond_sleep_validation_test) {
  kazunin_n_dining_philosophers_mpi::run_simulation(0.01, 0.1);
}

TEST(kazunin_n_dining_philosophers_mpi, 2000_millisecond_simulation_validation_test) {
  kazunin_n_dining_philosophers_mpi::run_simulation(2.0);
}
