#include <gtest/gtest.h>

#include <chrono>

#include "seq/konkov_i_task_dining_philosophers/include/ops_seq.hpp"

TEST(konkov_DiningPhilosophersSeq, PerformanceTest) {
  int philosopher_count = 10;
  int meals_per_philosopher = 100;

  konkov_i_task_dining_philosophers::DiningPhilosophers seq_task(philosopher_count, meals_per_philosopher);
  auto start = std::chrono::high_resolution_clock::now();
  seq_task.run();
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;
  ASSERT_LE(elapsed, 10.0);
}
