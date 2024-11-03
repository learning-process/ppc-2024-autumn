// Copyright 2023 Nasedkin Egor
#include <gtest/gtest.h>

#include <vector>

#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

TEST(Sequential_Operations_Seq, Test_Matrix_Column_Max) {
  std::vector<int> global_matrix;
  std::vector<int> global_max(3, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  const int rows = 4;
  const int cols = 3;
  global_matrix = nasedkin_e_matrix_column_max_value_seq::getRandomMatrix(rows, cols);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());

  nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSequential testSeqTaskSequential(taskDataSeq);
  ASSERT_EQ(testSeqTaskSequential.validation(), true);
  testSeqTaskSequential.pre_processing();
  testSeqTaskSequential.run();
  testSeqTaskSequential.post_processing();

  // Create data
  std::vector<int> reference_max(3, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeqRef = std::make_shared<ppc::core::TaskData>();
  taskDataSeqRef->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeqRef->inputs_count.emplace_back(global_matrix.size());
  taskDataSeqRef->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
  taskDataSeqRef->outputs_count.emplace_back(reference_max.size());

  // Create Task
  nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSequential testSeqTaskSequentialRef(taskDataSeqRef);
  ASSERT_EQ(testSeqTaskSequentialRef.validation(), true);
  testSeqTaskSequentialRef.pre_processing();
  testSeqTaskSequentialRef.run();
  testSeqTaskSequentialRef.post_processing();

  ASSERT_EQ(reference_max, global_max);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}