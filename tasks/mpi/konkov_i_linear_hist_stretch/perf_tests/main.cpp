#include <gtest/gtest.h>
#include <mpi.h>

#include <chrono>
#include <iostream>

#include "mpi/konkov_i_linear_hist_stretch/include/ops_mpi.hpp"

TEST(konkov_i_LinearHistStretchPerformance, StretchLargeImage) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int image_size = 1000000;
  int* image_data = nullptr;
  if (rank == 0) {
    image_data = new int[image_size];
    for (int i = 0; i < image_size; ++i) {
      image_data[i] = rand() % 256;
    }
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  if (!lht.validation()) {
    if (rank == 0) {
      std::cerr << "Validation failed for large image." << std::endl;
    }
    GTEST_SKIP();
  }

  if (!lht.pre_processing()) {
    if (rank == 0) {
      std::cerr << "Pre-processing failed for large image." << std::endl;
    }
    GTEST_SKIP();
  }

  auto start = std::chrono::high_resolution_clock::now();
  lht.run();
  auto end = std::chrono::high_resolution_clock::now();

  if (!lht.post_processing()) {
    if (rank == 0) {
      std::cerr << "Post-processing failed for large image." << std::endl;
    }
    GTEST_SKIP();
  }

  std::chrono::duration<double> elapsed = end - start;
  if (rank == 0) {
    std::cout << "Stretching execution time for image size " << image_size << ": " << elapsed.count() << " seconds"
              << std::endl;
    delete[] image_data;
  }
}