// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/gnitienko_k_sum_values_by_rows_matrix/include/ops_seq.hpp"

TEST(gnitienko_k_sum_row_seq, Test_rows_eq_cols) {
  const int rows = 10;
  const int cols = 10;

  // Create data
  std::vector<int> in(rows * cols, 0);
  for (int i = 0; i < rows; ++i) {
    in[i * cols] = i; 
  }
  std::vector<int> expect(rows, 0);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      expect[i] += in[i * cols + j]; 
    }
  }
  std::vector<int> out(rows, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(rows));
  taskDataSeq->inputs_count.emplace_back(static_cast<uint32_t>(cols));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(static_cast<uint32_t>(out.size()));

  // Create Task
  gnitienko_k_sum_row_seq::SumByRowSeq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expect, out);
}

TEST(gnitienko_k_sum_row_seq, Test_zero_values) {
  const int rows = 3;
  const int cols = 3;

  // Create data
  std::vector<int> in(rows * cols, 0);
  std::vector<int> expect(rows, 0);
  std::vector<int> out(rows, 0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count = {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count = {static_cast<uint32_t>(out.size())};

  // Create Task
  gnitienko_k_sum_row_seq::SumByRowSeq testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expect, out);
}

TEST(gnitienko_k_sum_row_seq, Test_arbitrary_values) {
  const int rows = 2;
  const int cols = 3;

  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6};
  std::vector<int> expect = {6, 15};  
  std::vector<int> out(rows, 0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count = {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count = {static_cast<uint32_t>(out.size())};

  // Create Task
  gnitienko_k_sum_row_seq::SumByRowSeq testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expect, out);
}

TEST(gnitienko_k_sum_row_seq, Test_negative_values) {
  const int rows = 2;
  const int cols = 2;

  // Create data
  std::vector<int> in = {-1, -2, -3, -4};
  std::vector<int> expect = {-3, -7};  
  std::vector<int> out(rows, 0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count = {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count = {static_cast<uint32_t>(out.size())};

  // Create Task
  gnitienko_k_sum_row_seq::SumByRowSeq testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expect, out);
}

TEST(gnitienko_k_sum_row_seq, Test_output_size) {
  const int rows = 5;
  const int cols = 3;

  // Create data
  std::vector<int> in(rows * cols, 1);
  std::vector<int> out(rows, 0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count = {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count = {static_cast<uint32_t>(out.size())};

  // Create Task
  gnitienko_k_sum_row_seq::SumByRowSeq testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out.size(), static_cast<size_t>(rows));
}

TEST(gnitienko_k_sum_row_seq, Test_output_element) {
  const int rows = 4;
  const int cols = 4;

  // Create data
  std::vector<int> in = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int> out(rows, 0);

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count = {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count = {static_cast<uint32_t>(out.size())};

  // Create Task
  gnitienko_k_sum_row_seq::SumByRowSeq testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(out[1], 26);  
}

TEST(gnitienko_k_sum_row_seq, Test_empty_input) {
  const int rows = 0;
  const int cols = 0;

  // Create data
  std::vector<int> in;
  std::vector<int> out;

  // Create TaskData
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count = {static_cast<uint32_t>(rows), static_cast<uint32_t>(cols)};
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count = {static_cast<uint32_t>(out.size())};

  // Create Task
  gnitienko_k_sum_row_seq::SumByRowSeq testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_TRUE(out.empty());
}
