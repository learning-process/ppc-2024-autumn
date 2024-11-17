#include <gtest/gtest.h>

#include <vector>

#include "seq/gnitienko_k_contrast_enhancement/include/ops_seq.hpp"

void run_test(std::vector<int>& img, double contrast_factor) {
  
  std::vector<int> out(img.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
  taskDataSeq->inputs_count.emplace_back(img.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq contrastEnhanceSeq(taskDataSeq);

  ASSERT_EQ(contrastEnhanceSeq.validation(), true);
  contrastEnhanceSeq.pre_processing();
  contrastEnhanceSeq.run();
  contrastEnhanceSeq.post_processing();

  std::vector<int> expected_out(img.size(), 0);
  for (size_t i = 0; i < img.size(); i++) expected_out[i] = std::clamp(static_cast<int>((img[i] - 128) * contrast_factor + 128), 0, 255);

  ASSERT_EQ(out, expected_out);
}

TEST(gnitienko_k_contrast_enhancement_seq, Test_Contrast_Enhancement_Grayscale) {
  std::vector<int> img = {0, 51, 102, 153, 204, 255, 124};
  double contrast_factor = 2.0;
  run_test(img, contrast_factor);
}

TEST(gnitienko_k_contrast_enhancement_seq, Test_Contrast_Enhancement_Color) {
  std::vector<int> img = {100, 150, 200, 50, 100, 150, 0, 0, 0, 255, 255, 255};
  double contrast_factor = 1.5;
  run_test(img, contrast_factor);
}

TEST(gnitienko_k_contrast_enhancement_seq, Test_Random_Grayscale_Image) {
  std::vector<int> img = gnitienko_k_contrast_enhancement_seq::getRandomVector(46);
  double contrast_factor = 3;
  run_test(img, contrast_factor);
}

TEST(gnitienko_k_contrast_enhancement_seq, Test_Random_Color_Image) {
  std::vector<int> img = gnitienko_k_contrast_enhancement_seq::getRandomVector(51);
  double contrast_factor = 3;
  run_test(img, contrast_factor);
}

TEST(gnitienko_k_contrast_enhancement_seq, Test_Color_Image_Pixel) {
  std::vector<int> img = {1, 2, 123, 4, 5};
  double contrast_factor = 3;
  std::vector<int> out(img.size(), 0);
  int pixel_3 = 113;

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
  taskDataSeq->inputs_count.emplace_back(img.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq contrastEnhanceSeq(taskDataSeq);

  ASSERT_EQ(contrastEnhanceSeq.validation(), true);
  contrastEnhanceSeq.pre_processing();
  contrastEnhanceSeq.run();
  contrastEnhanceSeq.post_processing();

  ASSERT_EQ(out[2], pixel_3);
}

TEST(gnitienko_k_contrast_enhancement_seq, Test_Empty_Image) {
  std::vector<int> img;
  double contrast_factor = 3;
  run_test(img, contrast_factor);
}

TEST(gnitienko_k_contrast_enhancement_seq, Test_One_Pixel_GrayScale_Image) {
  std::vector<int> img = {145};
  double contrast_factor = 1.5;
  run_test(img, contrast_factor);
}

TEST(gnitienko_k_contrast_enhancement_seq, Test_One_Pixel_Color_Image) {
  std::vector<int> img = {145, 134, 89};
  double contrast_factor = 1.5;
  run_test(img, contrast_factor);
}

TEST(gnitienko_k_contrast_enhancement_seq, Test_negative_values) {
  std::vector<int> img = {-145, -20, 15, -100, 234, -255, 45};
  double contrast_factor = 1.5;
  run_test(img, contrast_factor);
}


TEST(gnitienko_k_contrast_enhancement_seq, Test_incorrect_pixel) {
  std::vector<int> img = {750};
  double contrast_factor = 1.5;
  std::vector<int> out(img.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
  taskDataSeq->inputs_count.emplace_back(img.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq contrastEnhanceSeq(taskDataSeq);

  ASSERT_EQ(contrastEnhanceSeq.validation(), true);
  contrastEnhanceSeq.pre_processing();
  contrastEnhanceSeq.run();
  contrastEnhanceSeq.post_processing();

  ASSERT_EQ(out[0], 255);
}

TEST(gnitienko_k_contrast_enhancement_seq, Test_incorrect_pixel2) {
  std::vector<int> img = {750, -400, 1000};
  double contrast_factor = 1.5;
  std::vector<int> out(img.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(img.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&contrast_factor));
  taskDataSeq->inputs_count.emplace_back(img.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  gnitienko_k_contrast_enhancement_seq::ContrastEnhanceSeq contrastEnhanceSeq(taskDataSeq);

  ASSERT_EQ(contrastEnhanceSeq.validation(), true);
  contrastEnhanceSeq.pre_processing();
  contrastEnhanceSeq.run();
  contrastEnhanceSeq.post_processing();

  std::vector<int> expected_out = {255, 0, 255};

  ASSERT_EQ(out, expected_out);
}