#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

TEST(konkov_DiningPhilosophersMPI, PerformanceTest) {
  boost::mpi::timer timer;
  int philosopher_count = 10;
  int meals_per_philosopher = 100;

  konkov_i_task_dining_philosophers::DiningPhilosophers mpi_task(philosopher_count, meals_per_philosopher);
  timer.restart();
  mpi_task.run();
  double elapsed = timer.elapsed();

  ASSERT_LE(elapsed, 10.0);
}
