// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/zolotareva_a_smoothing_image/include/ops_seq.hpp"
namespace zolotareva_a_smoothing_image_seq {
std::vector<uint8_t> generateRandomImage(int height, int width) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  int size = height * width;
  std::vector<uint8_t> image(size);

  for (int i = 0; i < size; ++i) {
    image[i] = dis(gen);
  }

  return image;
}

void form(int h, int w) {
  unsigned int width = w;
  unsigned int height = h;
  std::vector<uint8_t> inputImage = generateRandomImage(height, width);
  std::vector<uint8_t> outputImage(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(inputImage.data());
  taskDataSeq->inputs_count.push_back(height);
  taskDataSeq->inputs_count.push_back(width);
  taskDataSeq->outputs.push_back(outputImage.data());
  taskDataSeq->outputs_count.push_back(outputImage.size());

  zolotareva_a_smoothing_image_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  EXPECT_EQ(height, taskDataSeq->inputs_count[0]);
}
}  // namespace zolotareva_a_smoothing_image_seq
TEST(zolotareva_a_smoothing_image_seq, Test_Image_random_1_1) { zolotareva_a_smoothing_image_seq::form(1, 1); };
TEST(zolotareva_a_smoothing_image_seq, Test_Image_random_3_3) { zolotareva_a_smoothing_image_seq::form(3, 3); };
TEST(zolotareva_a_smoothing_image_seq, Test_Image_random_59_26) { zolotareva_a_smoothing_image_seq::form(59, 26); };
TEST(zolotareva_a_smoothing_image_seq, Test_Image_random_25_50) { zolotareva_a_smoothing_image_seq::form(25, 59); };
TEST(zolotareva_a_smoothing_image_seq, Test_Image_random_100_100) { zolotareva_a_smoothing_image_seq::form(100, 100); };
TEST(zolotareva_a_smoothing_image_seq, BasicSmoothing) {
  unsigned short int width = 3;
  unsigned short int height = 3;
  std::vector<uint8_t> inputImage = {100, 100, 100, 100, 0, 100, 100, 100, 100};
  std::vector<uint8_t> outputImage(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(inputImage.data());
  taskDataSeq->inputs_count.push_back(height);
  taskDataSeq->inputs_count.push_back(width);
  taskDataSeq->outputs.push_back(outputImage.data());
  taskDataSeq->outputs_count.push_back(outputImage.size());

  zolotareva_a_smoothing_image_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  EXPECT_EQ(int(outputImage[width + 1]), 80);
}

TEST(zolotareva_a_smoothing_image_seq, OnePixelImage) {
  unsigned short int width = 1;
  std::vector<uint8_t> inputImage = {255};
  std::vector<uint8_t> outputImage(width * width);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(inputImage.data());
  taskDataSeq->inputs_count.push_back(width);
  taskDataSeq->inputs_count.push_back(width);
  taskDataSeq->outputs.push_back(outputImage.data());
  taskDataSeq->outputs_count.push_back(outputImage.size());

  zolotareva_a_smoothing_image_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  EXPECT_EQ(int(outputImage[0]), 255);
}

TEST(zolotareva_a_smoothing_image_seq, OneRowImage) {
  int width = 5;
  int height = 1;
  std::vector<uint8_t> inputImage = {0, 0, 255, 0, 0};
  std::vector<uint8_t> outputImage(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(inputImage.data());
  taskDataSeq->inputs_count.push_back(height);
  taskDataSeq->inputs_count.push_back(width);
  taskDataSeq->outputs.push_back(outputImage.data());
  taskDataSeq->outputs_count.push_back(outputImage.size());

  zolotareva_a_smoothing_image_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_TRUE(task.validation());
  task.pre_processing();
  task.run();
  task.post_processing();
  EXPECT_EQ(outputImage[2], 115);
}

TEST(zolotareva_a_smoothing_image_seq, InvalidInputSizes) {
  unsigned short int width = 5;
  unsigned short int height = 0;
  std::vector<uint8_t> inputImage(0);
  std::vector<uint8_t> outputImage(width * height);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(inputImage.data());
  taskDataSeq->inputs_count.push_back(height);
  taskDataSeq->inputs_count.push_back(width);
  taskDataSeq->outputs.push_back(outputImage.data());
  taskDataSeq->outputs_count.push_back(outputImage.size());

  zolotareva_a_smoothing_image_seq::TestTaskSequential task(taskDataSeq);
  EXPECT_FALSE(task.validation());
}

TEST(zolotareva_a_smoothing_image_seq, KernelCreation) {
  int radius = 1;
  int sigma = 1.0f;
  std::vector<float> kernel =
      zolotareva_a_smoothing_image_seq::TestTaskSequential::create_gaussian_kernel(radius, sigma);

  ASSERT_EQ(kernel.size(), size_t(3));
  float sum = 0.0f;
  for (float val : kernel) {
    sum += val;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-6);
}

TEST(zolotareva_a_smoothing_image_seq, ConvolutionCorrectness) {
  unsigned short int width = 3;
  unsigned short int height = 1;
  std::vector<uint8_t> inputImage = {0, 255, 0};
  std::vector<float> temp(width * height);
  std::vector<float> kernel = {0.25f, 0.5f, 0.25f};

  zolotareva_a_smoothing_image_seq::TestTaskSequential::convolve_rows(inputImage, height, width, kernel, temp);

  float fin_sum_0 = 0.25f * 0 + 0.5f * 0 + 0.25f * 255;
  float fin_sum_1 = 0.25f * 0 + 0.5f * 255 + 0.25f * 0;
  float fin_sum_2 = 0.25f * 255 + 0.5f * 0 + 0.25f * 0;

  EXPECT_NEAR(temp[0], fin_sum_0, 1e-6);
  EXPECT_NEAR(temp[1], fin_sum_1, 1e-6);
  EXPECT_NEAR(temp[2], fin_sum_2, 1e-6);
}
