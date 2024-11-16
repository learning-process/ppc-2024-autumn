// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>
//not example
#include <vector>

#include "seq/drozhdinov_d_gauss_vertical_scheme/include/ops_seq.hpp"

TEST(Sequential, EquationTest) {

  // Create data
  int rows = 2;
  int columns = 2;
  std::vector<double> matrix = {1, 0, 0, 1};
  std::vector<double> b = {1, 1};
  std::vector<double> expres(rows, 0);
  std::vector<double> res = {1, 1};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskDataSeq->inputs_count.emplace_back(matrix.size());
  taskDataSeq->inputs_count.emplace_back(b.size());
  taskDataSeq->inputs_count.emplace_back(columns);
  taskDataSeq->inputs_count.emplace_back(rows);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(expres.data()));
  taskDataSeq->outputs_count.emplace_back(expres.size());

  // Create Task
  drozhdinov_d_gauss_vertical_scheme_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, res);
}
/*
int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
*/
