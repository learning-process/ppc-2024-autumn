#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/konkov_i_linear_hist_stretch/include/ops_mpi.hpp"

TEST(konkov_i_LinearHistStretchTest, ValidImageData) {
  const int image_size = 100;
  int image_data[image_size];
  // Initialize image_data with sample values

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.validation());
  ASSERT_TRUE(lht.pre_processing());
  ASSERT_TRUE(lht.run());
  ASSERT_TRUE(lht.post_processing());
}

TEST(konkov_i_LinearHistStretchTest, InvalidImageData) {
  int image_size = 0;
  int* image_data = nullptr;

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_FALSE(lht.validation());
}
