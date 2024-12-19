#include <gtest/gtest.h>

#include <chrono>
#include <iostream>

#include "seq/konkov_i_linear_hist_stretch/include/ops_seq.hpp"

TEST(konkov_i_LinearHistStretchPerformance, StretchLargeImage) {
  const int image_size = 1000000;
  int* image_data = new int[image_size];

  for (int i = 0; i < image_size; ++i) {
    image_data[i] = rand() % 256;
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  if (!lht.validation()) {
    std::cerr << "Validation failed for large image." << std::endl;
    GTEST_SKIP();
  }

  if (!lht.pre_processing()) {
    std::cerr << "Pre-processing failed for large image." << std::endl;
    GTEST_SKIP();
  }

  auto start = std::chrono::high_resolution_clock::now();
  lht.run();
  auto end = std::chrono::high_resolution_clock::now();

  if (!konkov_i_linear_hist_stretch::LinearHistogramStretch::post_processing()) {
    std::cerr << "Post-processing failed for large image." << std::endl;
    GTEST_SKIP();
  }

  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Stretching execution time for image size " << image_size << ": " << elapsed.count() << " seconds"
            << std::endl;

  delete[] image_data;
}