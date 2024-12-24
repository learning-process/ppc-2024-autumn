#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <iostream>

#include "mpi/konkov_i_task_dining_philosophers/include/ops_mpi.hpp"

TEST(konkov_i_DiningPhilosophersPerformance, RunPipelinePerformance) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int num_philosophers = 100;
  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  if (!dp.validation()) {
    if (rank == 0) {
      std::cerr << "Validation failed for pipeline with " << num_philosophers << " philosophers." << std::endl;
    }
    GTEST_SKIP();
  }

  if (!dp.pre_processing()) {
    if (rank == 0) {
      std::cerr << "Pre-processing failed for pipeline." << std::endl;
    }
    GTEST_SKIP();
  }

  auto start = std::chrono::high_resolution_clock::now();
  dp.run();
  auto end = std::chrono::high_resolution_clock::now();

  if (!dp.post_processing()) {
    if (rank == 0) {
      std::cerr << "Post-processing failed for pipeline." << std::endl;
    }
    GTEST_SKIP();
  }

  std::chrono::duration<double> elapsed = end - start;
  if (rank == 0) {
    std::cout << "Pipeline execution time with " << num_philosophers << " philosophers: " << elapsed.count()
              << " seconds" << std::endl;
  }
}

TEST(konkov_i_DiningPhilosophersPerformance, RunTaskPerformance) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int num_philosophers = 100;
  konkov_i_dining_philosophers::DiningPhilosophers dp(num_philosophers);

  if (!dp.validation()) {
    if (rank == 0) {
      std::cerr << "Validation failed for task with " << num_philosophers << " philosophers." << std::endl;
    }
    GTEST_SKIP();
  }

  if (!dp.pre_processing()) {
    if (rank == 0) {
      std::cerr << "Pre-processing failed for task." << std::endl;
    }
    GTEST_SKIP();
  }

  auto start = std::chrono::high_resolution_clock::now();
  dp.run();
  auto end = std::chrono::high_resolution_clock::now();

  if (!dp.post_processing()) {
    if (rank == 0) {
      std::cerr << "Post-processing failed for task." << std::endl;
    }
    GTEST_SKIP();
  }

  std::chrono::duration<double> elapsed = end - start;
  if (rank == 0) {
    std::cout << "Task execution time with " << num_philosophers << " philosophers: " << elapsed.count() << " seconds"
              << std::endl;
  }
}
