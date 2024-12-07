#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>

#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

TEST(konkov_DiningPhilosophersMPI, BasicFunctionalityTest) {
  boost::mpi::communicator world;

  int philosopher_count = 5;
  int meals_per_philosopher = 3;

  if (world.rank() == 0) {
    std::vector<int> result(philosopher_count, 0);
    konkov_i_task_dining_philosophers::DiningPhilosophers mpi_task(philosopher_count, meals_per_philosopher);
    mpi_task.run();
    mpi_task.getResults(result);

    for (int meals : result) {
      ASSERT_EQ(meals, meals_per_philosopher);
    }
  }
}
