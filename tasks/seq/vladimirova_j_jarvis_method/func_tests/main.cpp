// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/vladimirova_j_jarvis_method/func_tests/test_val.cpp"
#include "seq/vladimirova_j_jarvis_method/include/ops_seq.hpp"

TEST(Sequential, Test_10_0) {
  const int n = 10;
  // Create data
  std::vector<int> in = data_10_0;
  std::vector<int> ans = ans_data_10_0;
  std::vector<int> out(ans.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, out);
}

TEST(Sequential, Test_10_1) {
  const int n = 10;
  // Create data
  std::vector<int> in = data_10_1;
  std::vector<int> out(ans_data_10_1.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans_data_10_1[0], out[0]);
}

TEST(Sequential, Test_10_2) {
  const int n = 10;
  // Create data
  std::vector<int> in = data_10_2;
  std::vector<int> out(ans_data_10_2.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans_data_10_2[0], out[0]);
}

TEST(Sequential, Test_5_0) {
  const int n = 5;
  // Create data
  std::vector<int> in = data_5_0;
  std::vector<int> out(ans_data_5_0.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans_data_5_0[0], out[0]);
}

TEST(Sequential, Test_5_1) {
  const int n = 5;
  // Create data
  std::vector<int> in = data_5_1;
  std::vector<int> out(ans_data_5_1.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans_data_5_1[0], out[0]);
}

TEST(Sequential, Test_5_2) {
  const int n = 5;
  // Create data
  std::vector<int> in = data_5_2;
  std::vector<int> out(ans_data_5_2.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans_data_5_2[0], out[0]);
}

TEST(Sequential, Test_data_5_empty) {
  const int n = 5;
  // Create data
  std::vector<int> in = data_5_empty;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(Sequential, Test_data_3_full) {
  const int n = 3;
  // Create data
  std::vector<int> in = data_3_full;
  std::vector<int> out(ans_data_3_full.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans_data_3_full[0], out[0]);
}