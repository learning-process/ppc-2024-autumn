// Copyright 2024 Anikin Maksim
#include <gtest/gtest.h>

#include <vector>

#include "seq/anikin_m_contrastscale/include/ops_seq.hpp"

using namespace anikin_m_contrastscale_seq;

TEST(anikin_m_contrastscale, increase_contrast) {
  const int count = 10;
  // Create data
  std::vector<RGB> in;
  in.reserve(count);
  float k = 1.5;
  std::vector<RGB> out;
  for (int i = 0; i < count; i++) {
    in.push_back(getrandomRGB());
  }
  out.resize(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ContrastScaleSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double inputContrast = getcontrast(in);
  double outputContrast = getcontrast(out);
  ASSERT_GE(outputContrast, inputContrast);
}

TEST(anikin_m_contrastscale, decrease_contrast) {
  const int count = 10;
  // Create data
  std::vector<RGB> in;
  in.reserve(count);
  float k = 0.5;
  std::vector<RGB> out;
  for (int i = 0; i < count; i++) {
    in.push_back(getrandomRGB());
  }
  out.resize(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ContrastScaleSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double inputContrast = getcontrast(in);
  double outputContrast = getcontrast(out);
  ASSERT_LE(outputContrast, inputContrast);
}

TEST(anikin_m_contrastscale, wrong_out_count) {
  const int count = 10;
  // Create data
  std::vector<RGB> in;
  in.reserve(count);
  float k = 0.5;
  std::vector<RGB> out;
  for (int i = 0; i < count; i++) {
    in.push_back(getrandomRGB());
  }
  out.resize(count + 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ContrastScaleSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(anikin_m_contrastscale, zero_image) {
  const int count = 0;
  // Create data
  std::vector<RGB> in;
  in.reserve(count);
  float k = 0.5;
  std::vector<RGB> out;
  for (int i = 0; i < count; i++) {
    in.push_back(getrandomRGB());
  }
  out.resize(count + 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&k));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  ContrastScaleSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}