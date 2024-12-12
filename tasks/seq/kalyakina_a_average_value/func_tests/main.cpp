// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/kalyakina_a_average_value/include/ops_seq.hpp"

std::vector<int> RandomVectorWithFixSum(int sum, const int &count) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> result_vector(count);
  for (int i = 0; i < count - 1; i++) {
    result_vector[i] = gen() % (std::min(sum, 255) - 1);
    sum -= result_vector[i];
  }
  result_vector[count - 1] = sum;
  return result_vector;
}

TEST(kalyakina_a_average_value_seq, Test_Avg_10) {
  const int count = 10;
  const int sum = 1000;
  const double expected_value = (double)(sum) / count;

  // Create data
  std::vector<int> in = RandomVectorWithFixSum(sum, count);
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential AverageValueTaskSequential(taskDataSeq);

  ASSERT_EQ(AverageValueTaskSequential.validation(), true);
  AverageValueTaskSequential.pre_processing();
  AverageValueTaskSequential.run();
  AverageValueTaskSequential.post_processing();

  ASSERT_DOUBLE_EQ(out[0], expected_value);
}

TEST(kalyakina_a_average_value_seq, Test_Avg_20) {
  const int count = 20;
  const int sum = 3500;
  const double expected_value = (double)sum / count;

  // Create data
  std::vector<int> in = RandomVectorWithFixSum(sum, count);
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential AverageValueTaskSequential(taskDataSeq);

  ASSERT_EQ(AverageValueTaskSequential.validation(), true);
  AverageValueTaskSequential.pre_processing();
  AverageValueTaskSequential.run();
  AverageValueTaskSequential.post_processing();

  ASSERT_DOUBLE_EQ(out[0], expected_value);
}

TEST(kalyakina_a_average_value_seq, Test_Avg_50) {
  const int count = 50;
  const int sum = 8000;
  const double expected_value = (double)sum / count;

  // Create data
  std::vector<int> in = RandomVectorWithFixSum(sum, count);
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential AverageValueTaskSequential(taskDataSeq);

  ASSERT_EQ(AverageValueTaskSequential.validation(), true);
  AverageValueTaskSequential.pre_processing();
  AverageValueTaskSequential.run();
  AverageValueTaskSequential.post_processing();

  ASSERT_DOUBLE_EQ(out[0], expected_value);
}

TEST(kalyakina_a_average_value_seq, Test_Avg_70) {
  const int count = 70;
  const int sum = 10000;
  const double expected_value = (double)sum / count;

  // Create data
  std::vector<int> in = RandomVectorWithFixSum(sum, count);
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential AverageValueTaskSequential(taskDataSeq);

  ASSERT_EQ(AverageValueTaskSequential.validation(), true);
  AverageValueTaskSequential.pre_processing();
  AverageValueTaskSequential.run();
  AverageValueTaskSequential.post_processing();

  ASSERT_DOUBLE_EQ(out[0], expected_value);
}

TEST(kalyakina_a_average_value_seq, Test_Avg_100) {
  const int count = 100;
  const int sum = 20000;
  const double expected_value = (double)sum / count;

  // Create data
  std::vector<int> in = RandomVectorWithFixSum(sum, count);
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  kalyakina_a_average_value_seq::FindingAverageOfVectorElementsTaskSequential AverageValueTaskSequential(taskDataSeq);

  ASSERT_EQ(AverageValueTaskSequential.validation(), true);
  AverageValueTaskSequential.pre_processing();
  AverageValueTaskSequential.run();
  AverageValueTaskSequential.post_processing();

  ASSERT_DOUBLE_EQ(out[0], expected_value);
}
