#include <gtest/gtest.h>

#include <vector>

#include "seq/dormidontov_e_highcontrast/include/egor_include.hpp"

namespace dormidontov_e_highcontrast_seq {
std::vector<int> generate_pic(int heigh, int width) {
  std::vector<int> tmp(heigh * width);
  for (int i = 0; i < heigh; ++i) {
    for (int j = 0; j < width; ++j) {
      tmp[i * width + j] = (i * width + j) % 6;
    }
  }
  return tmp;
}
std::vector<int> generate_answer(int heigh, int width) {
  std::vector<int> tmp(heigh * width);
  for (int i = 0; i < heigh; ++i) {
    for (int j = 0; j < width; ++j) {
      tmp[i * width + j] = ((i * width + j) % 6) * 51;
    }
  }
  return tmp;
}
}  // namespace dormidontov_e_highcontrast_seq

TEST(dormidontov_e_highcontrast_seq, test_min_values_by_columns_matrix_3x3) {
  int height = 3;
  int width = 3;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_highcontrast_seq::ContrastS ContrastS(taskDataSeq);
  std::vector<int> picture = dormidontov_e_highcontrast_seq::generate_pic(height, width);
  std::vector<int> exp_res = dormidontov_e_highcontrast_seq::generate_answer(height, width);
  std::vector<int> res_out(height * width);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));

  taskDataSeq->inputs_count.emplace_back(height * width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(ContrastS.validation(), true);
  ASSERT_TRUE(ContrastS.pre_processing());
  ASSERT_TRUE(ContrastS.run());
  ASSERT_TRUE(ContrastS.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_highcontrast_seq, test_min_values_by_columns_matrix_5x5) {
  int height = 5;
  int width = 5;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_highcontrast_seq::ContrastS ContrastS(taskDataSeq);
  std::vector<int> picture = dormidontov_e_highcontrast_seq::generate_pic(height, width);
  std::vector<int> exp_res = dormidontov_e_highcontrast_seq::generate_answer(height, width);
  std::vector<int> res_out(height * width);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));

  taskDataSeq->inputs_count.emplace_back(height * width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(ContrastS.validation(), true);
  ASSERT_TRUE(ContrastS.pre_processing());
  ASSERT_TRUE(ContrastS.run());
  ASSERT_TRUE(ContrastS.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_highcontrast_seq, test_min_values_by_columns_matrix_2x5) {
  int height = 2;
  int width = 5;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_highcontrast_seq::ContrastS ContrastS(taskDataSeq);
  std::vector<int> picture = dormidontov_e_highcontrast_seq::generate_pic(height, width);
  std::vector<int> exp_res = dormidontov_e_highcontrast_seq::generate_answer(height, width);
  std::vector<int> res_out(height * width);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));

  taskDataSeq->inputs_count.emplace_back(height * width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(ContrastS.validation(), true);
  ASSERT_TRUE(ContrastS.pre_processing());
  ASSERT_TRUE(ContrastS.run());
  ASSERT_TRUE(ContrastS.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_highcontrast_seq, test_min_values_by_columns_matrix_7x1) {
  int height = 7;
  int width = 1;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_highcontrast_seq::ContrastS ContrastS(taskDataSeq);
  std::vector<int> picture = dormidontov_e_highcontrast_seq::generate_pic(height, width);
  std::vector<int> exp_res = dormidontov_e_highcontrast_seq::generate_answer(height, width);
  std::vector<int> res_out(height * width);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));

  taskDataSeq->inputs_count.emplace_back(height * width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(ContrastS.validation(), true);
  ASSERT_TRUE(ContrastS.pre_processing());
  ASSERT_TRUE(ContrastS.run());
  ASSERT_TRUE(ContrastS.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_highcontrast_seq, test_min_values_by_columns_matrix_1x6) {
  int height = 1;
  int width = 6;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_highcontrast_seq::ContrastS ContrastS(taskDataSeq);
  std::vector<int> picture = dormidontov_e_highcontrast_seq::generate_pic(height, width);
  std::vector<int> exp_res = dormidontov_e_highcontrast_seq::generate_answer(height, width);
  std::vector<int> res_out(height * width);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));

  taskDataSeq->inputs_count.emplace_back(height * width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(ContrastS.validation(), true);
  ASSERT_TRUE(ContrastS.pre_processing());
  ASSERT_TRUE(ContrastS.run());
  ASSERT_TRUE(ContrastS.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_highcontrast_seq, test_min_values_by_columns_matrix_3000x3000) {
  int height = 3000;
  int width = 3000;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_highcontrast_seq::ContrastS ContrastS(taskDataSeq);
  std::vector<int> picture = dormidontov_e_highcontrast_seq::generate_pic(height, width);
  std::vector<int> exp_res = dormidontov_e_highcontrast_seq::generate_answer(height, width);
  std::vector<int> res_out(height * width);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));

  taskDataSeq->inputs_count.emplace_back(height * width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(ContrastS.validation(), true);
  ASSERT_TRUE(ContrastS.pre_processing());
  ASSERT_TRUE(ContrastS.run());
  ASSERT_TRUE(ContrastS.post_processing());
  ASSERT_EQ(res_out, exp_res);
}
TEST(dormidontov_e_highcontrast_seq, test_min_values_by_columns_matrix_1500x3000) {
  int height = 1500;
  int width = 3000;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_highcontrast_seq::ContrastS ContrastS(taskDataSeq);
  std::vector<int> picture = dormidontov_e_highcontrast_seq::generate_pic(height, width);
  std::vector<int> exp_res = dormidontov_e_highcontrast_seq::generate_answer(height, width);
  std::vector<int> res_out(height * width);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));

  taskDataSeq->inputs_count.emplace_back(height * width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(ContrastS.validation(), true);
  ASSERT_TRUE(ContrastS.pre_processing());
  ASSERT_TRUE(ContrastS.run());
  ASSERT_TRUE(ContrastS.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_highcontrast_seq, test_min_values_by_columns_matrix_1488x1488) {
  int height = 1488;
  int width = 1488;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_highcontrast_seq::ContrastS ContrastS(taskDataSeq);
  std::vector<int> picture = dormidontov_e_highcontrast_seq::generate_pic(height, width);
  std::vector<int> exp_res = dormidontov_e_highcontrast_seq::generate_answer(height, width);
  std::vector<int> res_out(height * width);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));

  taskDataSeq->inputs_count.emplace_back(height * width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(ContrastS.validation(), true);
  ASSERT_TRUE(ContrastS.pre_processing());
  ASSERT_TRUE(ContrastS.run());
  ASSERT_TRUE(ContrastS.post_processing());
  ASSERT_EQ(res_out, exp_res);
}