// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/poroshin_v_find_min_val_row_matrix/include/ops_seq.hpp"

TEST(poroshin_v_find_min_val_row_matrix_seq, find_min_10x10_matrix) {
  // Create data
  const int n = 10, m = 10;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  test->inputs_count.emplace_back(m);
  test->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m);

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int i = 0; i < m; i++) {
    ASSERT_EQ(result[i], INT_MIN);
  }
}

TEST(poroshin_v_find_min_val_row_matrix_seq, find_min_100x100_matrix) {
  // Create data
  const int n = 100, m = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  test->inputs_count.emplace_back(m);
  test->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m);

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int i = 0; i < m; i++) {
    ASSERT_EQ(result[i], INT_MIN);
  }
}

TEST(poroshin_v_find_min_val_row_matrix_seq, find_min_100x500_matrix) {
  // Create data
  const int n = 500, m = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  test->inputs_count.emplace_back(m);
  test->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m);

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int i = 0; i < m; i++) {
    ASSERT_EQ(result[i], INT_MIN);
  }
}

TEST(poroshin_v_find_min_val_row_matrix_seq, find_min_500x100_matrix) {
  // Create data
  const int n = 100, m = 500;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  test->inputs_count.emplace_back(m);
  test->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m);

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int i = 0; i < m; i++) {
    ASSERT_EQ(result[i], INT_MIN);
  }
}

TEST(poroshin_v_find_min_val_row_matrix_seq, find_min_5000x5000_matrix) {
  // Create data
  const int n = 5000, m = 5000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  test->inputs_count.emplace_back(m);
  test->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m);

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int i = 0; i < m; i++) {
    ASSERT_EQ(result[i], INT_MIN);
  }
}

TEST(poroshin_v_find_min_val_row_matrix_seq, validation_input_empty_100x100_matrix) {
  // Create data
  const int n = 100, m = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  // test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  test->inputs_count.emplace_back(m);
  test->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m);

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(poroshin_v_find_min_val_row_matrix_seq, validation_output_empty_100x100_matrix) {
  // Create data
  const int n = 100, m = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  test->inputs_count.emplace_back(m);
  test->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  // test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m);

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(poroshin_v_find_min_val_row_matrix_seq, validation_less_two_1_empty_100x100_matrix) {
  // Create data
  const int n = 100, m = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  test->inputs_count.emplace_back(m);
  // test->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m);

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(poroshin_v_find_min_val_row_matrix_seq, validation_less_two_2_empty_100x100_matrix) {
  // Create data
  const int n = 100, m = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  // test->inputs_count.emplace_back(m);
  test->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m);

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(poroshin_v_find_min_val_row_matrix_seq, validation_find_min_0x100_matrix) {
  // Create data
  const int n = 100, m = 0;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  test->inputs_count.emplace_back(m);
  test->inputs_count.emplace_back(n);
  std::vector<int> result(m);
  test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m);

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(poroshin_v_find_min_val_row_matrix_seq, validation_fails_on_invalid_output_size) {
  // Create data
  const int n = 100, m = 100;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> test = std::make_shared<ppc::core::TaskData>();
  poroshin_v_find_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(test);
  std::vector<int> tmp = poroshin_v_find_min_val_row_matrix_seq::gen(m, n);

  test->inputs.emplace_back(reinterpret_cast<uint8_t *>(tmp.data()));
  test->inputs_count.emplace_back(m);
  test->inputs_count.emplace_back(n);
  std::vector<int> result(m - 1);  // must be m
  test->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  test->outputs_count.emplace_back(m - 1);  // must be m

  ASSERT_EQ(testTaskSequential.validation(), false);
}