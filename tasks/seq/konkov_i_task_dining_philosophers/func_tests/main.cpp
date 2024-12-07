#include <gtest/gtest.h>

#include "seq/konkov_i_task_dining_philosophers/include/ops_seq.hpp"

TEST(konkov_DiningPhilosophersSeq, BasicFunctionalityTest) {
  int philosopher_count = 5;
  int meals_per_philosopher = 3;

  konkov_i_task_dining_philosophers::DiningPhilosophers seq_task(philosopher_count, meals_per_philosopher);
  seq_task.run();

  std::vector<int> results(philosopher_count, 0);
  seq_task.getResults(results);

  for (int meals : results) {
    ASSERT_EQ(meals, meals_per_philosopher);
  }
}
