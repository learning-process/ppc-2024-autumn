#include <gtest/gtest.h>

#include <vector>

#include "seq/dormidontov_e_min_value_by_columns_seq/include/ops_seq.hpp"

TEST(dormidontov_e_min_value_by_columns_seq, test_min_values_by_columns_matrix_3x3) {
  int rs = 3;
  int cs = 3;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_min_value_by_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<int> matrix = {11, 55, 33, 77, 99, 1010, 1111, 1212, 13};
  std::vector<int> res_out = {0, 0, 0};
  std::vector<int> exp_res = {11, 55, 13};

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));

  taskDataSeq->inputs_count.emplace_back(rs);
  taskDataSeq->inputs_count.emplace_back(cs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_min_value_by_columns_seq, test_min_values_by_columns_matrix_5x5) {
  int rs = 5;
  int cs = 5;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_min_value_by_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<int> matrix = {8,   5,    7,    8,    15,    17,   18,   1,     9,     10,    111,   127,  1388,
                             154, 1589, 1615, 1754, 18548, 1948, 2077, 21515, 22651, 23455, 24445, 25545};
  std::vector<int> res_out = {0, 0, 0, 0, 0};
  std::vector<int> exp_res = {8, 5, 1, 8, 10};

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));

  taskDataSeq->inputs_count.emplace_back(rs);
  taskDataSeq->inputs_count.emplace_back(cs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_min_value_by_columns_seq, test_min_values_by_columns_matrix_2x5) {
  int rs = 2;
  int cs = 5;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_min_value_by_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<int> matrix = {0, 0, 0, 22, 33, 44, 55, 66, 0, 0};
  std::vector<int> res_out = {0, 0, 0, 0, 0};
  std::vector<int> exp_res = {0, 0, 0, 0, 0};

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));

  taskDataSeq->inputs_count.emplace_back(rs);
  taskDataSeq->inputs_count.emplace_back(cs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_min_value_by_columns_seq, test_min_values_by_columns_matrix_7x1) {
  int rs = 7;
  int cs = 1;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_min_value_by_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<int> matrix = {11, 22, 33, 44, 55, 66, 77};
  std::vector<int> res_out = {0};
  std::vector<int> exp_res = {11};

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));

  taskDataSeq->inputs_count.emplace_back(rs);
  taskDataSeq->inputs_count.emplace_back(cs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_min_value_by_columns_seq, test_min_values_by_columns_matrix_1x5) {
  int rs = 1;
  int cs = 5;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_min_value_by_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<int> matrix = {11, 22, 33, 44, 55};
  std::vector<int> res_out = {0, 0, 0, 0, 0};
  std::vector<int> exp_res = {11, 22, 33, 44, 55};

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));

  taskDataSeq->inputs_count.emplace_back(rs);
  taskDataSeq->inputs_count.emplace_back(cs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_EQ(res_out, exp_res);
}

TEST(dormidontov_e_min_value_by_columns_seq, test_min_values_by_columns_matrix_3000x3000) {
  int rs = 3000;
  int cs = 3000;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_min_value_by_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<int> matrix(rs * cs);
  for (int i = 0; i < rs; ++i) {
    for (int j = 0; j < cs; ++j) {
      matrix[i * cs + j] = i * 1000 + j;
    }
  }
  std::vector<int> res_out(cs, 0);
  std::vector<int> exp_res(cs);
  for (int j = 0; j < cs; ++j) {
    exp_res[j] = j;
  }

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));

  taskDataSeq->inputs_count.emplace_back(rs);
  taskDataSeq->inputs_count.emplace_back(cs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_EQ(res_out, exp_res);
}
TEST(dormidontov_e_min_value_by_columns_seq, test_min_values_by_columns_matrix_1500x3000) {
  int rs = 1500;
  int cs = 3000;
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  dormidontov_e_min_value_by_columns_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<int> matrix(rs * cs);
  for (int i = 0; i < rs; ++i) {
    for (int j = 0; j < cs; ++j) {
      matrix[i * cs + j] = i * 1000 + j;
    }
  }
  std::vector<int> res_out(cs, 0);
  std::vector<int> exp_res(cs);
  for (int j = 0; j < cs; ++j) {
    exp_res[j] = j;
  }

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));

  taskDataSeq->inputs_count.emplace_back(rs);
  taskDataSeq->inputs_count.emplace_back(cs);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));

  taskDataSeq->outputs_count.emplace_back(res_out.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());
  ASSERT_EQ(res_out, exp_res);
}