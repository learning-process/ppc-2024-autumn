// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/lysov_i_simple_iteration_method/include/ops_seq.hpp"

TEST(lysov_i_simple_iteration_method_seq, SlaeIterationTask_IterationConvergence) {
  const int input_size = 3;

  std::vector<double> matrix = {4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 3.0};
  std::vector<double> g = {15.0, 10.0, 10.0};
  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(g.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->outputs_count.push_back(input_size);

  lysov_i_simple_iteration_method_seq::SlaeIterationTask slaeIterationTask(taskDataSeq);

  ASSERT_TRUE(slaeIterationTask.validation());
  slaeIterationTask.pre_processing();
  slaeIterationTask.run();
  slaeIterationTask.post_processing();

  const std::vector<double> expected_solution = {5.0, 5.0, 5.0};

  for (int i = 0; i < input_size; ++i) {
    EXPECT_NEAR(x[i], expected_solution[i], 1e-6);
  }
}

TEST(lysov_i_simple_iteration_method_seq, SlaeIterationTask_ZeroMatrixTest) {
  const int input_size = 3;
  std::vector<double> matrix = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> b = {15.0, 10.0, 10.0};
  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->outputs_count.push_back(input_size);

  lysov_i_simple_iteration_method_seq::SlaeIterationTask slaeIterationTask(taskDataSeq);

  ASSERT_FALSE(slaeIterationTask.validation());
}

TEST(lysov_i_simple_iteration_method_seq, SlaeIterationTask_InvalidDataTest) {
  const int input_size = 3;

  std::vector<double> matrix = {4.0, -1.0, 0.0, -1.0, 4.0, -1.0, 0.0, -1.0, 3.0, 1.0, 2.0, 3.0};
  std::vector<double> b = {15.0, 10.0, 10.0};
  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(4);
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->outputs_count.push_back(input_size);

  lysov_i_simple_iteration_method_seq::SlaeIterationTask slaeIterationTask(taskDataSeq);

  ASSERT_FALSE(slaeIterationTask.validation());
}

TEST(lysov_i_simple_iteration_method_seq, SlaeIterationTask_IterationConvergence_4x4) {
  const int input_size = 4;

  std::vector<double> matrix = {10, -1, 2, 0, -1, 11, -1, 3, 2, -1, 10, -1, 0, 3, -1, 8};

  std::vector<double> g = {6, 25, -11, 15};

  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(g.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->outputs_count.push_back(input_size);

  lysov_i_simple_iteration_method_seq::SlaeIterationTask slaeIterationTask(taskDataSeq);

  ASSERT_TRUE(slaeIterationTask.validation());
  slaeIterationTask.pre_processing();
  slaeIterationTask.run();
  slaeIterationTask.post_processing();

  std::vector<double> expected_solution = {1, 2, -1, 1};

  for (int i = 0; i < input_size; ++i) {
    EXPECT_NEAR(x[i], expected_solution[i], 1e-6);
  }
}

TEST(lysov_i_simple_iteration_method_seq, SlaeIterationTask_IterationConvergence_5x5) {
  const int input_size = 5;

  std::vector<double> matrix = {10, -1, 0, 0, 0, -1, 11, -1, 0, 0, 0, -1, 10, -1, 0, 0, 0, -1, 8, -1, 0, 0, 0, -1, 9};

  std::vector<double> g = {9, 7, 5, 3, 2};

  std::vector<double> x(input_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t *>(g.data()));
  taskDataSeq->inputs_count.push_back(input_size);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t *>(x.data()));
  taskDataSeq->outputs_count.push_back(input_size);

  lysov_i_simple_iteration_method_seq::SlaeIterationTask slaeIterationTask(taskDataSeq);

  ASSERT_TRUE(slaeIterationTask.validation());
  slaeIterationTask.pre_processing();
  slaeIterationTask.run();
  slaeIterationTask.post_processing();

  std::vector<double> expected_solution = {0.97823, 0.782296, 0.627023, 0.487932, 0.276437};

  for (int i = 0; i < input_size; ++i) {
    EXPECT_NEAR(x[i], expected_solution[i], 1e-5);
  }
}
