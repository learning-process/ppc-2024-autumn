// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/kondratev_ya_contrast_adjustment/include/ops_seq.hpp"

namespace kondratev_ya_contrast_adjustment_seq {
std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> genGradient(uint32_t side) {
  std::vector<kondratev_ya_contrast_adjustment_seq::Pixel> buff(side * side);
  auto step = (uint8_t)(255 / (2 * side - 1));

  for (uint32_t i = 0; i < side; i++) {
    for (uint32_t j = i; j < side; j++) {
      auto ind = i * side + j;
      auto ind2 = j * side + i;

      buff[ind] = step * (i + j + 1);
      buff[ind2] = step * (i + j + 1);
    }
  }
  return buff;
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
