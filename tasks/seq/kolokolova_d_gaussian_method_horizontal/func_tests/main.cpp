// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/kolokolova_d_gaussian_method_horizontal/include/ops_seq.hpp"

TEST(kolokolova_d_gaussian_method_horizontal_seq, Test_Guassian_Method1) {
  std::vector<int> input_coeff = {1, -1, 2, 1};
  std::vector<int> input_y = {-5, -7};
  std::vector<double> func_res(input_y.size(), 0);
  std::vector<double> ans = {-4, 1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
  taskDataSeq->inputs_count.emplace_back(input_coeff.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
  taskDataSeq->inputs_count.emplace_back(input_y.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  kolokolova_d_gaussian_method_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, func_res);
}

TEST(kolokolova_d_gaussian_method_horizontal_seq, Test_Guassian_Method2) {
  std::vector<int> input_coeff = {3, 2, -5, 2, -1, 3, 1, 2, -1};
  std::vector<int> input_y = {-1, 13, 9};
  std::vector<double> func_res(input_y.size(), 0);
  std::vector<double> ans = {3, 5, 4};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
  taskDataSeq->inputs_count.emplace_back(input_coeff.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
  taskDataSeq->inputs_count.emplace_back(input_y.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  kolokolova_d_gaussian_method_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, func_res);
}

TEST(kolokolova_d_gaussian_method_horizontal_seq, Test_Guassian_Method3) {
  std::vector<int> input_coeff = {1, 2, 3, 2, -1, 2, 1, 1, 5};
  std::vector<int> input_y = {1, 6, -1};
  std::vector<double> func_res(input_y.size(), 0);
  std::vector<double> ans = {4, 0, -1};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_coeff.data()));
  taskDataSeq->inputs_count.emplace_back(input_coeff.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_y.data()));
  taskDataSeq->inputs_count.emplace_back(input_y.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(func_res.data()));
  taskDataSeq->outputs_count.emplace_back(func_res.size());

  kolokolova_d_gaussian_method_horizontal_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, func_res);
}