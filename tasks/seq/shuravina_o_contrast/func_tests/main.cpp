#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

TEST(shuravina_o_contrast, Test_Contrast_Small) {
  const int count = 100;

  std::vector<uint8_t> in(count);
  std::vector<uint8_t> out(count, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  for (int i = 0; i < count; ++i) {
    in[i] = dis(gen);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_EQ(contrastTaskSequential.validation(), true);
  contrastTaskSequential.pre_processing();
  contrastTaskSequential.run();
  contrastTaskSequential.post_processing();

  uint8_t min_val = *std::min_element(in.begin(), in.end());
  uint8_t max_val = *std::max_element(in.begin(), in.end());
  for (int i = 0; i < count; ++i) {
    uint8_t expected = static_cast<uint8_t>((in[i] - min_val) * 255.0 / (max_val - min_val));
    ASSERT_EQ(out[i], expected);
  }
}
TEST(shuravina_o_contrast, Test_Contrast_Medium) {
  const int count = 10000;

  std::vector<uint8_t> in(count);
  std::vector<uint8_t> out(count, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  for (int i = 0; i < count; ++i) {
    in[i] = dis(gen);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_EQ(contrastTaskSequential.validation(), true);
  contrastTaskSequential.pre_processing();
  contrastTaskSequential.run();
  contrastTaskSequential.post_processing();

  uint8_t min_val = *std::min_element(in.begin(), in.end());
  uint8_t max_val = *std::max_element(in.begin(), in.end());
  for (int i = 0; i < count; ++i) {
    uint8_t expected = static_cast<uint8_t>((in[i] - min_val) * 255.0 / (max_val - min_val));
    ASSERT_EQ(out[i], expected);
  }
}
TEST(shuravina_o_contrast, Test_Contrast_Large) {
  const int count = 1000000;

  std::vector<uint8_t> in(count);
  std::vector<uint8_t> out(count, 0);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  for (int i = 0; i < count; ++i) {
    in[i] = dis(gen);
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_EQ(contrastTaskSequential.validation(), true);
  contrastTaskSequential.pre_processing();
  contrastTaskSequential.run();
  contrastTaskSequential.post_processing();

  uint8_t min_val = *std::min_element(in.begin(), in.end());
  uint8_t max_val = *std::max_element(in.begin(), in.end());
  for (int i = 0; i < count; ++i) {
    uint8_t expected = static_cast<uint8_t>((in[i] - min_val) * 255.0 / (max_val - min_val));
    ASSERT_EQ(out[i], expected);
  }
}

TEST(shuravina_o_contrast, Test_Contrast_MinMax) {
  const int count = 10000;

  std::vector<uint8_t> in(count);
  std::vector<uint8_t> out(count, 0);

  for (int i = 0; i < count; ++i) {
    in[i] = (i % 2 == 0) ? 0 : 255;
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_EQ(contrastTaskSequential.validation(), true);
  contrastTaskSequential.pre_processing();
  contrastTaskSequential.run();
  contrastTaskSequential.post_processing();

  for (int i = 0; i < count; ++i) {
    ASSERT_EQ(out[i], (i % 2 == 0) ? 0 : 255);
  }
}