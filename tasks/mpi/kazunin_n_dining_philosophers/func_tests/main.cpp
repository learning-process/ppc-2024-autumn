#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "core/task/include/task.hpp"
#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"

namespace kazunin_n_dining_philosophers_mpi {

bool is_valid(int eat_limit, int min_think_time, int max_think_time, int min_eat_time, int max_eat_time) {
  return eat_limit > 0 && min_think_time < max_think_time && min_eat_time < max_eat_time && max_think_time < 100 &&
         min_think_time > 0 && max_eat_time < 100 && min_eat_time > 0;
}

void start_test(int eat_limit = 3, int min_think_time = 10, int max_think_time = 20, int min_eat_time = 5,
                int max_eat_time = 10) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eat_limit));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&min_think_time));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_think_time));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&min_eat_time));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_eat_time));
  taskDataPar->inputs_count.emplace_back(1);

  auto taskParallel = std::make_shared<kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI>(taskDataPar);
  if (is_valid(eat_limit, min_think_time, max_think_time, min_eat_time, max_eat_time)) {
    EXPECT_TRUE(taskParallel->validation());
    taskParallel->pre_processing();
    taskParallel->run();
    taskParallel->post_processing();
  } else {
    EXPECT_FALSE(taskParallel->validation());
  }
  world.barrier();
}

}  // namespace kazunin_n_dining_philosophers_mpi

TEST(kazunin_n_dining_philosophers_mpi, defailt) { kazunin_n_dining_philosophers_mpi::start_test(); }

TEST(kazunin_n_dining_philosophers_mpi, default_eat_limit) { kazunin_n_dining_philosophers_mpi::start_test(4); }

TEST(kazunin_n_dining_philosophers_mpi, default_min_think_time) {
  kazunin_n_dining_philosophers_mpi::start_test(4, 10);
}

TEST(kazunin_n_dining_philosophers_mpi, default_max_think_time) {
  kazunin_n_dining_philosophers_mpi::start_test(4, 10, 15);
}

TEST(kazunin_n_dining_philosophers_mpi, default_min_eat_time) {
  kazunin_n_dining_philosophers_mpi::start_test(4, 10, 15, 10);
}

TEST(kazunin_n_dining_philosophers_mpi, default_max_eat_time) {
  kazunin_n_dining_philosophers_mpi::start_test(4, 10, 15, 10, 15);
}

TEST(kazunin_n_dining_philosophers_mpi, validation_test_min_eat_limit) {
  kazunin_n_dining_philosophers_mpi::start_test(0);
}

TEST(kazunin_n_dining_philosophers_mpi, validation_test_min_min_think_time) {
  kazunin_n_dining_philosophers_mpi::start_test(3, -20);
}

TEST(kazunin_n_dining_philosophers_mpi, validation_test_min_gt_max_think_time) {
  kazunin_n_dining_philosophers_mpi::start_test(3, 30, 20);
}

TEST(kazunin_n_dining_philosophers_mpi, validation_test_max_max_think_time) {
  kazunin_n_dining_philosophers_mpi::start_test(3, 20, 1000);
}

TEST(kazunin_n_dining_philosophers_mpi, validation_test_min_min_eat_time) {
  kazunin_n_dining_philosophers_mpi::start_test(3, 20, 40, -20);
}

TEST(kazunin_n_dining_philosophers_mpi, validation_test_min_gt_max_eat_time) {
  kazunin_n_dining_philosophers_mpi::start_test(3, 20, 40, 30, 20);
}

TEST(kazunin_n_dining_philosophers_mpi, validation_test_max_max_eat_time) {
  kazunin_n_dining_philosophers_mpi::start_test(3, 20, 40, 20, 1000);
}

TEST(kazunin_n_dining_philosophers_mpi, simulation_10_eat_limit) { kazunin_n_dining_philosophers_mpi::start_test(10); }

TEST(kazunin_n_dining_philosophers_mpi, simulation_20_eat_limit) {
  kazunin_n_dining_philosophers_mpi::start_test(20, 1, 2, 1, 2);
}

TEST(kazunin_n_dining_philosophers_mpi, simulation_30_eat_limit) {
  kazunin_n_dining_philosophers_mpi::start_test(30, 1, 2, 1, 2);
}
