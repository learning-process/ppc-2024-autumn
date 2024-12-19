#include <gtest/gtest.h>

#include "seq/konkov_i_linear_hist_stretch/include/ops_seq.hpp"

TEST(konkov_i_LinearHistStretchTest, ValidImageData) {
  const int image_size = 100;
  int image_data[image_size];

  // »нициализаци€ изображени€ с произвольными значени€ми
  for (int i = 0; i < image_size; ++i) {
    image_data[i] = rand() % 256;  // —лучайные значени€ от 0 до 255
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.validation());
  ASSERT_TRUE(lht.pre_processing());
  ASSERT_TRUE(lht.run());
  ASSERT_TRUE(lht.post_processing());

  // ѕроверка, что все значени€ пикселей наход€тс€ в диапазоне [0, 255]
  for (int i = 0; i < image_size; ++i) {
    EXPECT_GE(image_data[i], 0);
    EXPECT_LE(image_data[i], 255);
  }
}

TEST(konkov_i_LinearHistStretchTest, InvalidImageData) {
  int image_size = 0;
  int* image_data = nullptr;

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_FALSE(lht.validation());
}

TEST(konkov_i_LinearHistStretchTest, AllPixelsSameValueSeq) {
  const int image_size = 100;
  int image_data[image_size];

  // »нициализаци€ всех пикселей одинаковым значением
  for (int i = 0; i < image_size; ++i) {
    image_data[i] = 128;  // ¬се пиксели равны 128
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.validation());
  ASSERT_TRUE(lht.pre_processing());
  ASSERT_TRUE(lht.run());
  ASSERT_TRUE(lht.post_processing());

  for (int i = 0; i < image_size; ++i) {
    EXPECT_EQ(image_data[i], 128);
  }
}

TEST(konkov_i_LinearHistStretchTest, NegativeValuesSeq) {
  const int image_size = 100;
  int image_data[image_size];

  for (int i = 0; i < image_size; ++i) {
    image_data[i] = -100 + i;
  }

  konkov_i_linear_hist_stretch::LinearHistogramStretch lht(image_size, image_data);

  ASSERT_TRUE(lht.validation());
  ASSERT_TRUE(lht.pre_processing());
  ASSERT_TRUE(lht.run());
  ASSERT_TRUE(lht.post_processing());

  for (int i = 0; i < image_size; ++i) {
    EXPECT_GE(image_data[i], 0);
    EXPECT_LE(image_data[i], 255);
  }
}