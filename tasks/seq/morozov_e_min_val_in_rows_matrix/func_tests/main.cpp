#include <gtest/gtest.h>

#include <vector>

#include "seq/morozov_e_min_val_in_rows_matrix/include/ops_seq.hpp"

TEST(Sequential, Test_Validation_False) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  for (size_t i = 0; i < matrix.size(); ++i)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_FALSE(testMpiTaskSequential.validation());
}

TEST(Sequential, Test_Validation_True) {
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  std::vector<std::vector<int>> matrix = {{1, 1}, {2, 2}};
  std::vector<int> res = {1, 2};
  for (size_t i = 0; i < matrix.size(); ++i)
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(matrix[0].size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  taskDataSeq->outputs_count.emplace_back(res.size());
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_TRUE(testMpiTaskSequential.validation());
}
TEST(Sequential, Test_Main) {
  std::vector<std::vector<int>> matrix;
  const int n = 1000;
  const int m = 1000;
  std::vector<int> resSeq(n);
  std::vector<int> res(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  matrix = morozov_e_min_val_in_rows_matrix::getRandomMatrix(n, m);
  for (size_t i = 0; i < matrix.size(); ++i) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(matrix[i].data()));
  }

  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(resSeq.data()));
  taskDataSeq->outputs_count.emplace_back(resSeq.size());
  morozov_e_min_val_in_rows_matrix::TestTaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  // ASSERT_EQ(v, res2);
  for (int i = 0; i < n; ++i) {
    ASSERT_EQ(resSeq[i], -1);
  }
}