#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned/include/ops_seq.hpp"

namespace sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq {

void get_random_matrix(std::vector<int> &mat) {
  std::random_device dev;
  std::mt19937 gen(dev());

  for (size_t i = 0; i < mat.size(); ++i) {
    mat[i] = gen() % 1000 - 500;
  }
}
}  // namespace sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq, ValidationFailsForEmptyInputs) {
  // Create data
  std::vector<int> in;
  std::vector<int> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq, ValidInputsWithCorrectOutputSize) {
  // Create data
  std::vector<int> in_A(10, 0);
  std::vector<int> in_B(10, 0);
  std::vector<int> out(4, 0);
  std::vector<int> ans(4, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(2);
  taskDataSeq->inputs_count.emplace_back(5);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(5);
  taskDataSeq->inputs_count.emplace_back(2);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq, ValidationFailsForIncompatibleMatrixSizes) {
  // Create data
  std::vector<int> in_A(10, 0);
  std::vector<int> in_B(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(10);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(3);
  taskDataSeq->inputs_count.emplace_back(1);

  // Create Task
  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq, RandomInputMatrixWithValidDimensions) {
  // Create data

  std::vector<int> in_A(100, 0);
  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::get_random_matrix(in_A);
  std::vector<int> in_B(4, 0);
  std::vector<int> out(25, 0);
  std::vector<int> ans(25, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(25);
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq, CorrectResultForSquareMatrixMultiplication) {
  // Create data
  std::vector<int> in_A(16, 1);
  std::vector<int> in_B(16, 1);

  std::vector<int> out(16, 0);
  std::vector<int> ans(16, 4);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_A.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_B.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}