#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

namespace shuravina_o_contrast {

std::vector<uint8_t> generateRandomImage(size_t size) {
  std::vector<uint8_t> image(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(0, 255);

  for (size_t i = 0; i < size; ++i) {
    image[i] = static_cast<uint8_t>(distrib(gen));
  }

  return image;
}

}  // namespace shuravina_o_contrast

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_Empty_Input) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<uint8_t> input = {};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());

  std::vector<uint8_t> output(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_TRUE(contrastTaskSequential.validation());
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_Single_Element_Input) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<uint8_t> input = {50};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());

  std::vector<uint8_t> output(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_TRUE(contrastTaskSequential.validation());
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_Max_Values_Input) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<uint8_t> input = {255, 255, 255, 255, 255};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());

  std::vector<uint8_t> output(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_TRUE(contrastTaskSequential.validation());
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_Min_Values_Input) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<uint8_t> input = {0, 0, 0, 0, 0};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());

  std::vector<uint8_t> output(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_TRUE(contrastTaskSequential.validation());
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_100x32_Input) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<uint8_t> input = shuravina_o_contrast::generateRandomImage(100 * 32);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());

  std::vector<uint8_t> output(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_TRUE(contrastTaskSequential.validation());
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_64x23_Input) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<uint8_t> input = shuravina_o_contrast::generateRandomImage(64 * 23);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());

  std::vector<uint8_t> output(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_TRUE(contrastTaskSequential.validation());
}

TEST(shuravina_o_contrast, Test_Contrast_Enhancement_27x128_Input) {
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();

  std::vector<uint8_t> input = shuravina_o_contrast::generateRandomImage(27 * 128);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  taskDataSeq->inputs_count.emplace_back(input.size());

  std::vector<uint8_t> output(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.emplace_back(output.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_TRUE(contrastTaskSequential.validation());
}