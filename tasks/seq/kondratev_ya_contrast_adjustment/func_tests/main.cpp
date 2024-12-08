// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/kondratev_ya_contrast_adjustment/include/ops_seq.hpp"

namespace kondratev_ya_contrast_adjustment_seq {
std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> genGradient(uint32_t height, uint32_t width) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> buff(height * width);
  auto step = (uint8_t)(255 / (height + width - 1));

  for (uint32_t i = 0; i < height; i++) {
    for (uint32_t j = 0; j < width; j++) {
      buff[i * width + j] = step * (i + j + 1);
    }
  }
  return buff;
}

std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> genGradient(uint32_t side) {
  return kondratev_ya_contrast_adjustment_seq::genGradient(side, side);
}
}  // namespace kondratev_ya_contrast_adjustment_seq

TEST(kondratev_ya_contrast_adjustment_seq, gradient_test_increase) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> res;

  int side = 24;
  input = kondratev_ya_contrast_adjustment_seq::genGradient(side);
  res.resize(input.size());

  auto contrast = std::make_shared<double>(1.25);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kondratev_ya_contrast_adjustment_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  auto inputContrast = kondratev_ya_contrast_adjustment_seq::getContrast(input);
  auto resContrast = kondratev_ya_contrast_adjustment_seq::getContrast(res);

  ASSERT_GE(resContrast, inputContrast);
}

TEST(kondratev_ya_contrast_adjustment_seq, gradient_test_decrease) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> res;

  int side = 24;
  input = kondratev_ya_contrast_adjustment_seq::genGradient(side);
  res.resize(input.size());

  auto contrast = std::make_shared<double>(0.25);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kondratev_ya_contrast_adjustment_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  auto inputContrast = kondratev_ya_contrast_adjustment_seq::getContrast(input);
  auto resContrast = kondratev_ya_contrast_adjustment_seq::getContrast(res);

  ASSERT_LE(resContrast, inputContrast);
}

TEST(kondratev_ya_contrast_adjustment_seq, gradient_test_increase_with_non_equal_sizes) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> res;

  int height = 5;
  int width = 10;
  input = kondratev_ya_contrast_adjustment_seq::genGradient(height, width);
  res.resize(input.size());

  auto contrast = std::make_shared<double>(1.25);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kondratev_ya_contrast_adjustment_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  auto inputContrast = kondratev_ya_contrast_adjustment_seq::getContrast(input);
  auto resContrast = kondratev_ya_contrast_adjustment_seq::getContrast(res);

  ASSERT_GE(resContrast, inputContrast);
}

TEST(kondratev_ya_contrast_adjustment_seq, gradient_test_decrease_with_non_equal_sizes) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> res;

  int height = 13;
  int width = 7;
  input = kondratev_ya_contrast_adjustment_seq::genGradient(height, width);
  res.resize(input.size());

  auto contrast = std::make_shared<double>(0.25);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kondratev_ya_contrast_adjustment_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.validation();
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  auto inputContrast = kondratev_ya_contrast_adjustment_seq::getContrast(input);
  auto resContrast = kondratev_ya_contrast_adjustment_seq::getContrast(res);

  ASSERT_LE(resContrast, inputContrast);
}

TEST(kondratev_ya_contrast_adjustment_seq, bad_contrast) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> res;

  int height = 13;
  int width = 7;
  input = kondratev_ya_contrast_adjustment_seq::genGradient(height, width);
  res.resize(input.size());

  auto contrast = std::make_shared<double>(-0.25);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
  taskDataSeq->inputs_count.emplace_back(input.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());

  kondratev_ya_contrast_adjustment_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  bool ret = testTaskSequential.validation();
  ASSERT_EQ(ret, false);
}

TEST(kondratev_ya_contrast_adjustment_seq, different_sizes) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  int side = 24;
  input = kondratev_ya_contrast_adjustment_seq::genGradient(side);
  contrast = std::make_shared<double>(0.25);

  res.resize(input.size() + 1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
  taskDataPar->inputs_count.emplace_back(input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataPar->outputs_count.emplace_back(res.size());

  kondratev_ya_contrast_adjustment_seq::TestTaskSequential TestTaskSequential(taskDataPar);
  bool ret = TestTaskSequential.validation();
  ASSERT_EQ(ret, false);
}

TEST(kondratev_ya_contrast_adjustment_seq, no_out) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  int side = 24;
  input = kondratev_ya_contrast_adjustment_seq::genGradient(side);
  contrast = std::make_shared<double>(0.25);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
  taskDataPar->inputs_count.emplace_back(input.size());
  taskDataPar->outputs_count.emplace_back(input.size());

  kondratev_ya_contrast_adjustment_seq::TestTaskSequential TestTaskSequential(taskDataPar);
  bool ret = TestTaskSequential.validation();
  ASSERT_EQ(ret, false);
}

TEST(kondratev_ya_contrast_adjustment_seq, no_in) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<double> contrast;

  int side = 24;
  input = kondratev_ya_contrast_adjustment_seq::genGradient(side);
  contrast = std::make_shared<double>(0.25);
  res.resize(input.size());

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(contrast.get()));
  taskDataPar->inputs_count.emplace_back(input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataPar->outputs_count.emplace_back(res.size());

  kondratev_ya_contrast_adjustment_seq::TestTaskSequential TestTaskSequential(taskDataPar);
  bool ret = TestTaskSequential.validation();
  ASSERT_EQ(ret, false);
}

TEST(kondratev_ya_contrast_adjustment_seq, no_constast) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> input;
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> res;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  int side = 24;
  input = kondratev_ya_contrast_adjustment_seq::genGradient(side);
  res.resize(input.size());

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  taskDataPar->inputs_count.emplace_back(input.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataPar->outputs_count.emplace_back(res.size());

  kondratev_ya_contrast_adjustment_seq::TestTaskSequential TestTaskSequential(taskDataPar);
  bool ret = TestTaskSequential.validation();
  ASSERT_EQ(ret, false);
}
