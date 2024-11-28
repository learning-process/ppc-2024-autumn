#include <gtest/gtest.h>

#include <vector>

#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

TEST(Sequential_Contrast, Test_Contrast_10) {
  std::vector<uint8_t> input_vec(10, 128);
  std::vector<uint8_t> output_vec(10, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vec.data()));
  taskDataSeq->outputs_count.emplace_back(output_vec.size());

  shuravina_o_contrast::ContrastSequential contrastTask(taskDataSeq);
  ASSERT_EQ(contrastTask.validation(), true);
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();

  for (int i = 0; i < static_cast<int>(output_vec.size()); ++i) {
    ASSERT_EQ(output_vec[i], 255);
  }
}

TEST(Sequential_Contrast, Test_Contrast_20) {
  std::vector<uint8_t> input_vec(20, 64);
  std::vector<uint8_t> output_vec(20, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vec.data()));
  taskDataSeq->outputs_count.emplace_back(output_vec.size());

  shuravina_o_contrast::ContrastSequential contrastTask(taskDataSeq);
  ASSERT_EQ(contrastTask.validation(), true);
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();

  for (int i = 0; i < static_cast<int>(output_vec.size()); ++i) {
    ASSERT_EQ(output_vec[i], 255);
  }
}

TEST(Sequential_Contrast, Test_Contrast_30) {
  std::vector<uint8_t> input_vec(30, 32);
  std::vector<uint8_t> output_vec(30, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vec.data()));
  taskDataSeq->outputs_count.emplace_back(output_vec.size());

  shuravina_o_contrast::ContrastSequential contrastTask(taskDataSeq);
  ASSERT_EQ(contrastTask.validation(), true);
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();

  for (int i = 0; i < static_cast<int>(output_vec.size()); ++i) {
    ASSERT_EQ(output_vec[i], 255);
  }
}

TEST(Sequential_Contrast, Test_Contrast_40) {
  std::vector<uint8_t> input_vec(40, 16);
  std::vector<uint8_t> output_vec(40, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vec.data()));
  taskDataSeq->inputs_count.emplace_back(input_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vec.data()));
  taskDataSeq->outputs_count.emplace_back(output_vec.size());

  shuravina_o_contrast::ContrastSequential contrastTask(taskDataSeq);
  ASSERT_EQ(contrastTask.validation(), true);
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();

  for (int i = 0; i < static_cast<int>(output_vec.size()); ++i) {
    ASSERT_EQ(output_vec[i], 255);
  }
}