#include <gtest/gtest.h>

#include <vector>

#include "seq/gromov_a_sum_of_vector_elements/include/ops_seq.hpp"

TEST(gromov_a_sum_of_vector_elements_seq, Test_Sum_30) {
  const int count = 30;

  // Create data
  std::vector<int> in(1, count);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  gromov_a_sum_of_vector_elements_seq::SumOfVector sumOfVector(taskDataSeq);
  ASSERT_EQ(sumOfVector.validation(), true);
  sumOfVector.pre_processing();
  sumOfVector.run();
  sumOfVector.post_processing();
  ASSERT_EQ(count, out[0]);
}

TEST(gromov_a_sum_of_vector_elements_seq, Test_Max_Element) {
  std::vector<int> in = {4, 1, 3, 2, 5};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  gromov_a_sum_of_vector_elements_seq::SumOfVector sumOfVector(taskDataSeq);
  ASSERT_EQ(sumOfVector.validation(), true);
  sumOfVector.pre_processing();
  sumOfVector.run();
  sumOfVector.post_processing();
  ASSERT_EQ(5, *std::max_element(in.begin(), in.end()));
}

TEST(gromov_a_sum_of_vector_elements_seq, Test_Min_Element) {
  std::vector<int> in = {1, 3, 5, 2, 4};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  gromov_a_sum_of_vector_elements_seq::SumOfVector sumOfVector(taskDataSeq);
  ASSERT_EQ(sumOfVector.validation(), true);
  sumOfVector.pre_processing();
  sumOfVector.run();
  sumOfVector.post_processing();
  ASSERT_EQ(1, *std::min_element(in.begin(), in.end()));
}

TEST(gromov_a_sum_of_vector_elements_seq, Test_SumTwoElements) {
  const int a = 2;
  const int b = 3;
  std::vector<int> in = {a, b};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  gromov_a_sum_of_vector_elements_seq::SumOfVector sumOfVector(taskDataSeq);
  ASSERT_EQ(sumOfVector.validation(), true);
  sumOfVector.pre_processing();
  sumOfVector.run();
  sumOfVector.post_processing();
  ASSERT_EQ(a + b, out[0]);
}

TEST(gromov_a_sum_of_vector_elements_seq, Test_BigVector) {
  const int count = 10000;

  // Create data
  std::vector<int> in(count, 1);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  gromov_a_sum_of_vector_elements_seq::SumOfVector sumOfVector(taskDataSeq);
  ASSERT_EQ(sumOfVector.validation(), true);
  sumOfVector.pre_processing();
  sumOfVector.run();
  sumOfVector.post_processing();
  ASSERT_EQ(count, out[0]);
}

TEST(gromov_a_sum_of_vector_elements_seq, Test_Sum_70) {
  const int count = 70;

  // Create data
  std::vector<int> in(1, count);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  gromov_a_sum_of_vector_elements_seq::SumOfVector sumOfVector(taskDataSeq);
  ASSERT_EQ(sumOfVector.validation(), true);
  sumOfVector.pre_processing();
  sumOfVector.run();
  sumOfVector.post_processing();
  ASSERT_EQ(count, out[0]);
}

TEST(gromov_a_sum_of_vector_elements_seq, Test_Max2_Element) {
  std::vector<int> in = {9, 15, 43, 22, 11};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  gromov_a_sum_of_vector_elements_seq::SumOfVector sumOfVector(taskDataSeq);
  ASSERT_EQ(sumOfVector.validation(), true);
  sumOfVector.pre_processing();
  sumOfVector.run();
  sumOfVector.post_processing();
  ASSERT_EQ(43, *std::max_element(in.begin(), in.end()));
}

TEST(gromov_a_sum_of_vector_elements_seq, Test_BigVector2) {
  const int count = 30000;

  // Create data
  std::vector<int> in(count, 1);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  gromov_a_sum_of_vector_elements_seq::SumOfVector sumOfVector(taskDataSeq);
  ASSERT_EQ(sumOfVector.validation(), true);
  sumOfVector.pre_processing();
  sumOfVector.run();
  sumOfVector.post_processing();
  ASSERT_EQ(count, out[0]);
}

TEST(gromov_a_sum_of_vector_elements_seq, Test_SumTwoElements2) {
  const int a = 105;
  const int b = 113;
  std::vector<int> in = {a, b};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  gromov_a_sum_of_vector_elements_seq::SumOfVector sumOfVector(taskDataSeq);
  ASSERT_EQ(sumOfVector.validation(), true);
  sumOfVector.pre_processing();
  sumOfVector.run();
  sumOfVector.post_processing();
  ASSERT_EQ(a + b, out[0]);
}

TEST(gromov_a_sum_of_vector_elements_seq, Test_BigVector3) {
  const int count = 500000;

  // Create data
  std::vector<int> in(count, 1);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  gromov_a_sum_of_vector_elements_seq::SumOfVector sumOfVector(taskDataSeq);
  ASSERT_EQ(sumOfVector.validation(), true);
  sumOfVector.pre_processing();
  sumOfVector.run();
  sumOfVector.post_processing();
  ASSERT_EQ(count, out[0]);
}