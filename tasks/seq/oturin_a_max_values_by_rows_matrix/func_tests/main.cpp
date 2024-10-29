#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "seq/oturin_a_max_values_by_rows_matrix/include/ops_seq.hpp"

TEST(oturin_a_max_values_by_rows_matrix_seq_functest, Test_Max_5_5) {
  size_t n = 5;
  size_t m = 5;

  // Create data
  std::vector<int> in(n * m);
  std::vector<int> out(m, 0);
  std::vector<int> maxes(m);

  std::iota(std::begin(in), std::end(in), 1);
  for (size_t i = 0; i < m; i++) maxes[i] = (i + 1) * n;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  oturin_a_max_values_by_rows_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(maxes, out);
}

TEST(oturin_a_max_values_by_rows_matrix_seq_functest, Test_Max_10_5) {
  size_t n = 10;
  size_t m = 5;

  // Create data
  std::vector<int> in(n * m);
  std::vector<int> out(m, 0);
  std::vector<int> maxes(m);

  std::iota(std::begin(in), std::end(in), 1);
  for (size_t i = 0; i < m; i++) maxes[i] = (i + 1) * n;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  oturin_a_max_values_by_rows_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(maxes, out);
}

TEST(oturin_a_max_values_by_rows_matrix_seq_functest, Test_Max_5_10) {
  size_t n = 5;
  size_t m = 10;

  // Create data
  std::vector<int> in(n * m);
  std::vector<int> out(m, 0);
  std::vector<int> maxes(m);

  std::iota(std::begin(in), std::end(in), 1);
  for (size_t i = 0; i < m; i++) maxes[i] = (i + 1) * n;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  oturin_a_max_values_by_rows_matrix_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(maxes, out);
}
