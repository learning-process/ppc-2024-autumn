// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/khovansky_d_max_of_vector_elements/include/ops_seq.hpp"

std::vector<int> khovansky_d_max_of_vector_elements_seq::GetRandomVector(int sz, int left, int right) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(sz);
  for (int i = 0; i < sz; i++) {
    v[i] = gen() % (1 + right - left) + left;
  }
  return v;
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_10_Positive_Numbers) {
  const int count = 10;
  const int left = 0;
  const int right = 1000;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_10_Negative_Numbers) {
  const int count = 10;
  const int left = -1000;
  const int right = -1;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_20_Positive_Numbers) {
  const int count = 20;
  const int left = 0;
  const int right = 1000;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_20_Negative_Numbers) {
  const int count = 20;
  const int left = -1000;
  const int right = -1;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_50_Positive_Numbers) {
  const int count = 50;
  const int left = 0;
  const int right = 100;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_50_Negative_Numbers) {
  const int count = 50;
  const int left = -1000;
  const int right = -1;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_70_Positive_Numbers) {
  const int count = 70;
  const int left = 0;
  const int right = 100;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_70_Negative_Numbers) {
  const int count = 70;
  const int left = -1000;
  const int right = -1;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_100_Positive_Numbers) {
  const int count = 100;
  const int left = 0;
  const int right = 100;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_100_Negative_Numbers) {
  const int count = 100;
  const int left = -1000;
  const int right = -1;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_1000_Positive_Numbers) {
  const int count = 1000;
  const int left = 0;
  const int right = 100;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_1000_Negative_Numbers) {
  const int count = 1000;
  const int left = -1000;
  const int right = -1;
  // Create data
  std::vector<int> in = khovansky_d_max_of_vector_elements_seq::GetRandomVector(count, left, right);
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  int ex = *std::max_element(in.begin(), in.end());
  ASSERT_EQ(ex, out[0]);
}

TEST(khovansky_d_max_of_vector_elements_seq, Test_Max_Empty_Vector) {
  
  // Create data
  std::vector<int> in = {};
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  khovansky_d_max_of_vector_elements_seq::MaxOfVectorSeq MaxOfVectorSeq(taskDataSeq);
  ASSERT_EQ(MaxOfVectorSeq.validation(), true);
  MaxOfVectorSeq.pre_processing();
  MaxOfVectorSeq.run();
  MaxOfVectorSeq.post_processing();
  
  ASSERT_EQ(0, out[0]);
}