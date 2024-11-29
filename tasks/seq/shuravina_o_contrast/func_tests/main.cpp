#include <gtest/gtest.h>

#include <vector>

#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

TEST(Sequential_Contrast, Test_Contrast_10) {
  const int count = 10;

  std::vector<uint8_t> in(count, 128);
  std::vector<uint8_t> out(count, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_EQ(contrastTaskSequential.validation(), true);
  contrastTaskSequential.pre_processing();
  contrastTaskSequential.run();
  contrastTaskSequential.post_processing();
  ASSERT_EQ(out, std::vector<uint8_t>(count, 128));
}

TEST(Sequential_Contrast, Test_Contrast_20) {
  const int count = 20;

  std::vector<uint8_t> in(count, 64);
  std::vector<uint8_t> out(count, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  shuravina_o_contrast::ContrastTaskSequential contrastTaskSequential(taskDataSeq);
  ASSERT_EQ(contrastTaskSequential.validation(), true);
  contrastTaskSequential.pre_processing();
  contrastTaskSequential.run();
  contrastTaskSequential.post_processing();
  ASSERT_EQ(out, std::vector<uint8_t>(count, 128));
}