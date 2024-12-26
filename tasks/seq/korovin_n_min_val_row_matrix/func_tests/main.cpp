// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/korovin_n_min_val_row_matrix/include/ops_seq.hpp"

TEST(korovin_n_min_val_row_matrix_seq, find_min_val_in_row_10x10_matrix) {
  const int rows = 10;
  const int cols = 10;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> v_res(rows, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int i = 0; i < rows; i++) {
    ASSERT_EQ(v_res[i], INT_MIN);
  }
}

TEST(korovin_n_min_val_row_matrix_seq, find_min_val_in_row_100x100_matrix) {
  const int rows = 100;
  const int cols = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> v_res(rows, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int i = 0; i < rows; i++) {
    ASSERT_EQ(v_res[i], INT_MIN);
  }
}

TEST(korovin_n_min_val_row_matrix_seq, find_min_val_in_row_100x500_matrix) {
  const int rows = 100;
  const int cols = 500;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> v_res(rows, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int i = 0; i < rows; i++) {
    ASSERT_EQ(v_res[i], INT_MIN);
  }
}

TEST(korovin_n_min_val_row_matrix_seq, find_min_val_in_row_3000x3000_matrix) {
  const int rows = 3000;
  const int cols = 3000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> v_res(rows, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int i = 0; i < rows; i++) {
    ASSERT_EQ(v_res[i], INT_MIN);
  }
}

TEST(korovin_n_min_val_row_matrix_seq, validation_input_empty_100x100_matrix) {
  const int rows = 100;
  const int cols = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> v_res(rows, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(korovin_n_min_val_row_matrix_seq, validation_output_empty_100x100_matrix) {
  const int rows = 100;
  const int cols = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> v_res(rows, 0);
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(korovin_n_min_val_row_matrix_seq, validation_less_two_100x100_matrix) {
  const int rows = 100;
  const int cols = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);

  std::vector<int> v_res(rows, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(korovin_n_min_val_row_matrix_seq, validation_less_two_cols_100x100_matrix) {
  const int rows = 100;
  const int cols = 100;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> v_res(rows, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(korovin_n_min_val_row_matrix_seq, validation_find_min_val_in_row_0x10_matrix) {
  const int rows = 0;
  const int cols = 10;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> v_res(rows, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(korovin_n_min_val_row_matrix_seq, validation_find_min_val_in_row_10x10_cols_0_matrix) {
  const int rows = 10;
  const int cols = 10;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(0);

  std::vector<int> v_res(rows, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(korovin_n_min_val_row_matrix_seq, validation_fails_on_invalid_output_size) {
  const int rows = 10;
  const int cols = 10;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  korovin_n_min_val_row_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  std::vector<std::vector<int>> matrix_rnd =
      korovin_n_min_val_row_matrix_seq::TestTaskSequential::generate_rnd_matrix(rows, cols);

  for (auto& row : matrix_rnd) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> v_res(rows - 1, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(v_res.data()));
  taskDataSeq->outputs_count.emplace_back(v_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}
