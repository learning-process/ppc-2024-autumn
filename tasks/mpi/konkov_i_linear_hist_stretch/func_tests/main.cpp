#include <gtest/gtest.h>
#include <mpi.h>

#include "mpi/konkov_i_linear_hist_stretch/include/ops_mpi.hpp"

TEST(konkov_i_LinearHistStretchTest, ValidImageData) {
  const int image_size = 100;
  int image_data[image_size];
  for (int i = 0; i < image_size; ++i) {
    image_data[i] = rand() % 256;  // Random values between 0 and 255
  }

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

TEST(konkov_i_LinearHistStretchTest, AllPixelsSameValueMPI) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int image_size = 100;
  int* image_data = nullptr;

  if (rank == 0) {
    image_data = new int[image_size];
    for (int i = 0; i < image_size; ++i) {
      image_data[i] = 128;
    }
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  if (rank == 0) {
    ASSERT_TRUE(lht.validation());
  }

  ASSERT_TRUE(lht.pre_processing());

  ASSERT_TRUE(lht.run());

  ASSERT_TRUE(lht.post_processing());

  if (rank == 0) {
    for (int i = 0; i < image_size; ++i) {
      EXPECT_EQ(image_data[i], 128);
    }
    delete[] image_data;
  }
}

TEST(konkov_i_LinearHistStretchTest, NegativeValuesMPI) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int image_size = 100;
  int* image_data = nullptr;

  if (rank == 0) {
    image_data = new int[image_size];
    for (int i = 0; i < image_size; ++i) {
      image_data[i] = -100 + i;
    }
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.validation());
  ASSERT_TRUE(lht.pre_processing());
  ASSERT_TRUE(lht.run());
  ASSERT_TRUE(lht.post_processing());

  if (rank == 0) {
    for (int i = 0; i < image_size; ++i) {
      EXPECT_GE(image_data[i], 0);
      EXPECT_LE(image_data[i], 255);
    }
    delete[] image_data;
  }
}