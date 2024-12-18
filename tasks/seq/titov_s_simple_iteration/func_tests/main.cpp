// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/titov_s_simple_iteration/include/ops_seq.hpp"

TEST(titov_s_simple_iteration_seq, Test_Simple_Iteration_2_not_DiagonallyDominant) {
  std::vector<std::vector<float>> input_matrix = {{10.0f, 5.0f, 20.0f}, {6.0f, -2.0f, 2.0f}};

  float epsilon = 0.001f;

  std::vector<float> expected_result = {1.0f, 2.0f};

  std::vector<float> output(expected_result.size(), 0.0f);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (const auto& row : input_matrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
  }

  taskDataSeq->inputs_count.push_back(input_matrix.size());
  taskDataSeq->inputs_count.push_back(input_matrix[0].size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.push_back(output.size());

  titov_s_simple_iteration_seq::SimpleIterationSequential SimpleIterationSequential(taskDataSeq);

  ASSERT_TRUE(SimpleIterationSequential.validation());
  ASSERT_FALSE(SimpleIterationSequential.pre_processing());
}

TEST(titov_s_simple_iteration_seq, Test_Simple_Iteration_4) {
  std::vector<std::vector<float>> input_matrix = {{20.9f, 1.2f, 2.1f, 0.9f, 21.7f},
                                                  {1.2f, 21.2f, 1.5f, 2.5f, 27.46f},
                                                  {2.1f, 1.5f, 19.8f, 1.3f, 28.76f},
                                                  {0.9f, 2.5f, 1.3f, 32.1f, 49.72f}};

  float epsilon = 0.001f;
  std::vector<float> expected_result = {0.8f, 1.0f, 1.2f, 1.4f};
  std::vector<float> output(expected_result.size(), 0.0f);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  for (const auto& row : input_matrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
  }

  taskDataSeq->inputs_count.push_back(input_matrix.size());
  taskDataSeq->inputs_count.push_back(input_matrix[0].size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.push_back(output.size());

  titov_s_simple_iteration_seq::SimpleIterationSequential SimpleIterationSequential(taskDataSeq);

  ASSERT_TRUE(SimpleIterationSequential.validation());
  ASSERT_TRUE(SimpleIterationSequential.pre_processing());
  ASSERT_TRUE(SimpleIterationSequential.run());
  ASSERT_TRUE(SimpleIterationSequential.post_processing());

  for (size_t i = 0; i < expected_result.size(); ++i) {
    ASSERT_NEAR(expected_result[i], output[i], epsilon);
  }
}

TEST(titov_s_simple_iteration_seq, Test_Simple_Iteration_4_0) {
  std::vector<std::vector<float>> input_matrix = {{10.0f, -1.0f, 2.0f, 0.0f, 6.0f},
                                                  {-1.0f, 11.0f, -1.0f, 3.0f, 25.0f},
                                                  {2.0f, -1.0f, 10.0f, -1.0f, -11.0f},
                                                  {0.0f, 3.0f, -1.0f, 8.0f, 15.0f}};

  float epsilon = 0.001f;

  std::vector<float> expected_result = {1.0f, 2.0f, -1.0f, 1.0f};

  std::vector<float> output(expected_result.size(), 0.0f);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (const auto& row : input_matrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
  }

  taskDataSeq->inputs_count.push_back(input_matrix.size());
  taskDataSeq->inputs_count.push_back(input_matrix[0].size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.push_back(output.size());

  titov_s_simple_iteration_seq::SimpleIterationSequential SimpleIterationSequential(taskDataSeq);

  ASSERT_TRUE(SimpleIterationSequential.validation());
  ASSERT_TRUE(SimpleIterationSequential.pre_processing());
  ASSERT_TRUE(SimpleIterationSequential.run());
  ASSERT_TRUE(SimpleIterationSequential.post_processing());

  for (size_t i = 0; i < expected_result.size(); ++i) {
    ASSERT_NEAR(expected_result[i], output[i], epsilon);
  }
}

TEST(titov_s_simple_iteration_seq, Test_Simple_Iteration_2_simple) {
  std::vector<std::vector<float>> input_matrix = {{10.0f, 5.0f, 20.0f}, {6.0f, -15.0f, -24.0f}};

  float epsilon = 0.01f;

  std::vector<float> expected_result = {1.0f, 2.0f};

  std::vector<float> output(expected_result.size(), 0.0f);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (const auto& row : input_matrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
  }

  taskDataSeq->inputs_count.push_back(input_matrix.size());
  taskDataSeq->inputs_count.push_back(input_matrix[0].size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.push_back(output.size());

  titov_s_simple_iteration_seq::SimpleIterationSequential SimpleIterationSequential(taskDataSeq);

  ASSERT_TRUE(SimpleIterationSequential.validation());
  ASSERT_TRUE(SimpleIterationSequential.pre_processing());
  ASSERT_TRUE(SimpleIterationSequential.run());
  ASSERT_TRUE(SimpleIterationSequential.post_processing());
  for (size_t i = 0; i < expected_result.size(); ++i) {
    ASSERT_NEAR(expected_result[i], output[i], epsilon);
  }
}

TEST(titov_s_simple_iteration_seq, Test_Simple_Iteration_2_0) {
  std::vector<std::vector<float>> input_matrix = {{10.0f, 0.0f, 1.0f}, {0.0f, 5.0f, 2.0f}};

  float epsilon = 0.01f;

  std::vector<float> expected_result = {0.1f, 0.4f};

  std::vector<float> output(expected_result.size(), 0.0f);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (const auto& row : input_matrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
  }

  taskDataSeq->inputs_count.push_back(input_matrix.size());
  taskDataSeq->inputs_count.push_back(input_matrix[0].size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.push_back(output.size());

  titov_s_simple_iteration_seq::SimpleIterationSequential SimpleIterationSequential(taskDataSeq);

  ASSERT_TRUE(SimpleIterationSequential.validation());
  ASSERT_TRUE(SimpleIterationSequential.pre_processing());
  ASSERT_TRUE(SimpleIterationSequential.run());
  ASSERT_TRUE(SimpleIterationSequential.post_processing());
  for (size_t i = 0; i < expected_result.size(); ++i) {
    ASSERT_NEAR(expected_result[i], output[i], epsilon);
  }
}

TEST(titov_s_simple_iteration_seq, Test_Simple_Iteration_1) {
  std::vector<std::vector<float>> input_matrix = {{10.0f, 1.0f}};

  float epsilon = 0.01f;

  std::vector<float> expected_result = {0.1f};

  std::vector<float> output(expected_result.size(), 0.0f);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (const auto& row : input_matrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
  }

  taskDataSeq->inputs_count.push_back(input_matrix.size());
  taskDataSeq->inputs_count.push_back(input_matrix[0].size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.push_back(output.size());

  titov_s_simple_iteration_seq::SimpleIterationSequential SimpleIterationSequential(taskDataSeq);

  ASSERT_TRUE(SimpleIterationSequential.validation());
  ASSERT_TRUE(SimpleIterationSequential.pre_processing());
  ASSERT_TRUE(SimpleIterationSequential.run());
  ASSERT_TRUE(SimpleIterationSequential.post_processing());

  for (size_t i = 0; i < expected_result.size(); ++i) {
    ASSERT_NEAR(expected_result[i], output[i], epsilon);
  }
}

TEST(titov_s_simple_iteration_seq, Test_Simple_Iteration_Inputs_Empty) {
  std::vector<std::vector<float>> input_matrix = {{}};
  float epsilon = 0.01f;
  std::vector<float> expected_result = {0.1f};
  std::vector<float> output(expected_result.size(), 0.0f);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (const auto& row : input_matrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
  }
  taskDataSeq->inputs_count.push_back(input_matrix.size());
  taskDataSeq->inputs_count.push_back(input_matrix[0].size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.push_back(output.size());
  titov_s_simple_iteration_seq::SimpleIterationSequential SimpleIterationSequential(taskDataSeq);
  ASSERT_FALSE(SimpleIterationSequential.validation());
}

TEST(titov_s_simple_iteration_seq, Test_Simple_Iteration_Non_Square_Matrix) {
  std::vector<std::vector<float>> input_matrix = {{10.0f, 3.0f, 1.0f}};
  float* epsilon = nullptr;
  std::vector<float> expected_result = {0.1f, 0.4f};
  std::vector<float> output(expected_result.size(), 0.0f);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (const auto& row : input_matrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
  }
  taskDataSeq->inputs_count.push_back(input_matrix.size());
  taskDataSeq->inputs_count.push_back(input_matrix[0].size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.push_back(output.size());
  titov_s_simple_iteration_seq::SimpleIterationSequential SimpleIterationSequential(taskDataSeq);
  ASSERT_FALSE(SimpleIterationSequential.validation());
}

TEST(titov_s_simple_iteration_seq, Test_Simple_Iteration_Wrong_Epsilon) {
  std::vector<std::vector<float>> input_matrix = {{10.0f, 1.0f}};
  float epsilon = 1.4f;
  std::vector<float> expected_result = {0.1f};
  std::vector<float> output(expected_result.size(), 0.0f);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (const auto& row : input_matrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
  }
  taskDataSeq->inputs_count.push_back(input_matrix.size());
  taskDataSeq->inputs_count.push_back(input_matrix[0].size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.push_back(output.size());
  titov_s_simple_iteration_seq::SimpleIterationSequential SimpleIterationSequential(taskDataSeq);
  ASSERT_FALSE(SimpleIterationSequential.validation());
}

TEST(titov_s_simple_iteration_seq, Test_Simple_Iteration_Wrong_Output) {
  std::vector<std::vector<float>> input_matrix = {{10.0f, 1.0f}};
  float epsilon = 0.4f;
  std::vector<float> expected_result = {};
  std::vector<float> output(expected_result.size(), 0.0f);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  for (const auto& row : input_matrix) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<float*>(row.data())));
  }
  taskDataSeq->inputs_count.push_back(input_matrix.size());
  taskDataSeq->inputs_count.push_back(input_matrix[0].size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&epsilon));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataSeq->outputs_count.push_back(output.size());
  titov_s_simple_iteration_seq::SimpleIterationSequential SimpleIterationSequential(taskDataSeq);
  ASSERT_FALSE(SimpleIterationSequential.validation());
}