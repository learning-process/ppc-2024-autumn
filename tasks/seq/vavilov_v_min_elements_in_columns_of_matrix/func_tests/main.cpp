#include <gtest/gtest.h>

#include <vector>

#include "seq/vavilov_v_min_elements_in_columns_of_matrix/include/ops_seq.hpp"

TEST(vavilov_v_min_elements_in_columns_of_matrix_seq, find_min_elem_in_col_400x500_matr) {
  const int rows = 400;
  const int cols = 500;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matr =
      vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential::generate_rand_matr(rows, cols);

  for (auto& row : matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> vec_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
  taskDataSeq->outputs_count.emplace_back(vec_res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (size_t i = 0; i < input_[0].size(); i++) {
    int min = input_[0][i];
    for (size_t j = 1; j < input_.size(); j++) {
      if (input_[j][i] < min) {
        min = input_[j][i];
      }
    ASSERT_EQ(vec_res[j], min);
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_seq, find_min_elem_in_col_3000x3000_matr) {
  const int rows = 3000;
  const int cols = 3000;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matr =
      vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential::generate_rand_matr(rows, cols);

  for (auto& row : matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> vec_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
  taskDataSeq->outputs_count.emplace_back(vec_res.size());

  ASSERT_EQ(testTaskSequential.validation(), true);
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  for (int j = 0; j < cols; j++) {
    ASSERT_EQ(vec_res[j], INT_MIN);
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_seq, validation_input_empty_10x10_matr) {
  const int rows = 10;
  const int cols = 10;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matr =
      vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential::generate_rand_matr(rows, cols);

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> vec_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
  taskDataSeq->outputs_count.emplace_back(vec_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_seq, validation_output_empty_10x10_matr) {
  const int rows = 10;
  const int cols = 10;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matr =
      vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential::generate_rand_matr(rows, cols);

  for (auto& row : matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> vec_res(cols, 0);
  taskDataSeq->outputs_count.emplace_back(vec_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_seq, validation_find_min_elem_in_col_10x0_matr) {
  const int rows = 10;
  const int cols = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  std::vector<std::vector<int>> matr =
      vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential::generate_rand_matr(rows, cols);

  for (auto& row : matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> vec_res(cols, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
  taskDataSeq->outputs_count.emplace_back(vec_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_seq, validation_fails_on_invalid_output_of_size) {
  const int rows = 10;
  const int cols = 10;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);

  std::vector<std::vector<int>> matr =
      vavilov_v_min_elements_in_columns_of_matrix_seq::TestTaskSequential::generate_rand_matr(rows, cols);

  for (auto& row : matr) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
  }

  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->inputs_count.emplace_back(cols);

  std::vector<int> vec_res(cols - 1, 0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(vec_res.data()));
  taskDataSeq->outputs_count.emplace_back(vec_res.size());

  ASSERT_EQ(testTaskSequential.validation(), false);
}
