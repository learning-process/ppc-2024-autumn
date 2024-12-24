// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/vladimirova_j_jarvis_method/func_tests/test_val.cpp"
#include "seq/vladimirova_j_jarvis_method/include/ops_seq.hpp"

TEST(Sequential, Test_10_0) {
  const int n = 10;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_10_0;
  std::vector<int> ans = vladimirova_j_jarvis_method_seq::ans_data_10_0;
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
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_10_1;
  std::vector<int> out(vladimirova_j_jarvis_method_seq::ans_data_10_1.size());

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

  ASSERT_EQ(vladimirova_j_jarvis_method_seq::ans_data_10_1[0], out[0]);
}

TEST(Sequential, Test_10_2) {
  const int n = 10;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_10_2;
  std::vector<int> out(vladimirova_j_jarvis_method_seq::ans_data_10_2.size());

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

  ASSERT_EQ(vladimirova_j_jarvis_method_seq::ans_data_10_2[0], out[0]);
}

TEST(Sequential, Test_5_0) {
  const int n = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_5_0;
  std::vector<int> out(vladimirova_j_jarvis_method_seq::ans_data_5_0.size());

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

  ASSERT_EQ(vladimirova_j_jarvis_method_seq::ans_data_5_0[0], out[0]);
}

TEST(Sequential, Test_5_1) {
  const int n = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_5_1;
  std::vector<int> out(vladimirova_j_jarvis_method_seq::ans_data_5_1.size());

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

  ASSERT_EQ(vladimirova_j_jarvis_method_seq::ans_data_5_1[0], out[0]);
}

TEST(Sequential, Test_5_2) {
  const int n = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_5_2;
  std::vector<int> out(vladimirova_j_jarvis_method_seq::ans_data_5_2.size());

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

  ASSERT_EQ(vladimirova_j_jarvis_method_seq::ans_data_5_2[0], out[0]);
}

TEST(Sequential, Test_data_5_empty) {
  const int n = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_5_empty;
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
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_3_full;
  std::vector<int> out(vladimirova_j_jarvis_method_seq::ans_data_3_full.size());

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

  ASSERT_EQ(vladimirova_j_jarvis_method_seq::ans_data_3_full[0], out[0]);
}

TEST(Sequential, Test_10_5_0) {
  const int row = 5;
  const int col = 10;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_10_5_0;
  std::vector<int> ans = vladimirova_j_jarvis_method_seq::ans_data_10_5_0;
  std::vector<int> out(ans.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(5);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, out);
}

TEST(Sequential, Test_20_0) {
  const int n = 20;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_20_0;
  std::vector<int> ans = vladimirova_j_jarvis_method_seq::ans_data_20_0;
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

TEST(Sequential, Test_20_1) {
  const int n = 20;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_20_1;
  std::vector<int> ans = vladimirova_j_jarvis_method_seq::ans_data_20_1;
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

TEST(Sequential, Test_2_5_0) {
  const int row = 2;
  const int col = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_2_5_0;
  std::vector<int> ans = vladimirova_j_jarvis_method_seq::ans_data_2_5_0;
  std::vector<int> out(ans.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_EQ(ans, out);
}

TEST(Sequential, Test_data_one_row) {
  const int row = 1;
  const int col = 2;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_5_empty;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(Sequential, Test_data_one_col) {
  const int col = 1;
  const int row = 3;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_5_empty;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(Sequential, Test_data_5_one_line_0) {
  const int col = 5;
  const int row = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_5_one_line_0;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(Sequential, Test_data_5_one_line_1) {
  const int col = 5;
  const int row = 5;
  // Create data
  std::vector<int> in = vladimirova_j_jarvis_method_seq::data_5_one_line_1;
  std::vector<int> out(1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(row);
  taskDataSeq->inputs_count.emplace_back(col);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vladimirova_j_jarvis_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}
