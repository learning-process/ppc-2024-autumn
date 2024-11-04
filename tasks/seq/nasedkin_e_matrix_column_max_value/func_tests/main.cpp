// Copyright 2023 Nasedkin Egor
#include <gtest/gtest.h>
#include <vector>
#include "seq/nasedkin_e_matrix_column_max_value/include/ops_seq.hpp"

TEST(Sequential_Operations_Seq, Test_Matrix_Column_Max) {
  std::vector<int> global_matrix;
  std::vector<int> global_max(3, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  const int rows = 6, cols = 3;
  global_matrix = nasedkin_e_matrix_column_max_value_seq::getRandomMatrix(rows, cols);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
  taskDataSeq->inputs_count.emplace_back(global_matrix.size());
  taskDataSeq->inputs_count.emplace_back(cols);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
  taskDataSeq->outputs_count.emplace_back(global_max.size());

  nasedkin_e_matrix_column_max_value_seq::MatrixColumnMaxSeq matrixColumnMaxSeq(taskDataSeq);
  ASSERT_EQ(matrixColumnMaxSeq.validation(), true);
  matrixColumnMaxSeq.pre_processing();
  matrixColumnMaxSeq.run();
  matrixColumnMaxSeq.post_processing();

  std::vector<int> reference_max(3, 0);
  for (int col = 0; col < 3; col++) {
    reference_max[col] = global_matrix[col];
    for (int row = 1; row < 6; row++) {
      if (global_matrix[row * 3 + col] > reference_max[col]) {
        reference_max[col] = global_matrix[row * 3 + col];
      }
    }
  }
  ASSERT_EQ(reference_max, global_max);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}