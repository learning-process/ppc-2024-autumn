#include <gtest/gtest.h>

#include <vector>

#include "seq/morozov_e_mult_sparse_matrix/include/ops_seq.hpp"
TEST(morozov_e_mult_sparse_matrix, Test_Validation_columnsA_notEqual_rowsB_1) {
  std::vector<std::vector<double>> matrixA = {{0, 2}, {1, 0}, {0, 4}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(matrixA.size());
  taskData->inputs_count.emplace_back(matrixA[0].size());
  taskData->inputs_count.emplace_back(dA.size());
  taskData->inputs_count.emplace_back(row_indA.size());
  taskData->inputs_count.emplace_back(col_indA.size());

  taskData->inputs_count.emplace_back(matrixB.size());
  taskData->inputs_count.emplace_back(matrixB[0].size());
  taskData->inputs_count.emplace_back(dB.size());
  taskData->inputs_count.emplace_back(row_indB.size());
  taskData->inputs_count.emplace_back(col_indB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  for (int i = 0; i < out.size(); ++i) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
  }
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs_count.emplace_back(out[0].size());

  morozov_e_mult_sparse_matrix::TestTaskSequential testTaskSequential(taskData);
  ASSERT_FALSE(testTaskSequential.validation());
}
TEST(morozov_e_mult_sparse_matrix, Test_Validation_columnsA_notEqual_rowsB_2) {
  std::vector<std::vector<double>> matrixA = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(matrixA.size());
  taskData->inputs_count.emplace_back(matrixA[0].size());
  taskData->inputs_count.emplace_back(dA.size());
  taskData->inputs_count.emplace_back(row_indA.size());
  taskData->inputs_count.emplace_back(col_indA.size());

  taskData->inputs_count.emplace_back(matrixB.size());
  taskData->inputs_count.emplace_back(matrixB[0].size());
  taskData->inputs_count.emplace_back(dB.size());
  taskData->inputs_count.emplace_back(row_indB.size());
  taskData->inputs_count.emplace_back(col_indB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  for (int i = 0; i < out.size(); ++i) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
  }
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs_count.emplace_back(out[0].size());

  morozov_e_mult_sparse_matrix::TestTaskSequential testTaskSequential(taskData);
  ASSERT_FALSE(testTaskSequential.validation());
}
TEST(morozov_e_mult_sparse_matrix, Test_Validation_columnsAns_notEqual_columnsB) {
  std::vector<std::vector<double>> matrixA = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(matrixA.size());
  taskData->inputs_count.emplace_back(matrixA[0].size());
  taskData->inputs_count.emplace_back(dA.size());
  taskData->inputs_count.emplace_back(row_indA.size());
  taskData->inputs_count.emplace_back(col_indA.size());

  taskData->inputs_count.emplace_back(matrixB.size());
  taskData->inputs_count.emplace_back(matrixB[0].size());
  taskData->inputs_count.emplace_back(dB.size());
  taskData->inputs_count.emplace_back(row_indB.size());
  taskData->inputs_count.emplace_back(col_indB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  for (int i = 0; i < out.size(); ++i) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
  }
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs_count.emplace_back(out[0].size() + 1);

  morozov_e_mult_sparse_matrix::TestTaskSequential testTaskSequential(taskData);
  ASSERT_FALSE(testTaskSequential.validation());
}
TEST(morozov_e_mult_sparse_matrix, Test_Validation_rowssAns_notEqual_rowsB) {
  std::vector<std::vector<double>> matrixA = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(matrixA.size());
  taskData->inputs_count.emplace_back(matrixA[0].size());
  taskData->inputs_count.emplace_back(dA.size());
  taskData->inputs_count.emplace_back(row_indA.size());
  taskData->inputs_count.emplace_back(col_indA.size());

  taskData->inputs_count.emplace_back(matrixB.size());
  taskData->inputs_count.emplace_back(matrixB[0].size());
  taskData->inputs_count.emplace_back(dB.size());
  taskData->inputs_count.emplace_back(row_indB.size());
  taskData->inputs_count.emplace_back(col_indB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  for (int i = 0; i < out.size(); ++i) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
  }
  taskData->outputs_count.emplace_back(out.size() + 1);
  taskData->outputs_count.emplace_back(out[0].size());

  morozov_e_mult_sparse_matrix::TestTaskSequential testTaskSequential(taskData);
  ASSERT_FALSE(testTaskSequential.validation());
}
TEST(morozov_e_mult_sparse_matrix, Test_Main) {
  std::vector<std::vector<double>> matrixA = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<std::vector<double>> matrixB = {{0, 2, 0}, {1, 0, 3}, {0, 4, 0}};
  std::vector<double> dA;
  std::vector<int> row_indA;
  std::vector<int> col_indA;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixA, dA, row_indA, col_indA);
  std::vector<double> dB;
  std::vector<int> row_indB;
  std::vector<int> col_indB;
  morozov_e_mult_sparse_matrix::convertToCCS(matrixB, dB, row_indB, col_indB);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.emplace_back(matrixA.size());
  taskData->inputs_count.emplace_back(matrixA[0].size());
  taskData->inputs_count.emplace_back(dA.size());
  taskData->inputs_count.emplace_back(row_indA.size());
  taskData->inputs_count.emplace_back(col_indA.size());

  taskData->inputs_count.emplace_back(matrixB.size());
  taskData->inputs_count.emplace_back(matrixB[0].size());
  taskData->inputs_count.emplace_back(dB.size());
  taskData->inputs_count.emplace_back(row_indB.size());
  taskData->inputs_count.emplace_back(col_indB.size());
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indA.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(dB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(row_indB.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(col_indB.data()));
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  for (int i = 0; i < out.size(); ++i) {
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out[i].data()));
  }
  taskData->outputs_count.emplace_back(out.size());
  taskData->outputs_count.emplace_back(out[0].size());

  morozov_e_mult_sparse_matrix::TestTaskSequential testTaskSequential(taskData);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<std::vector<double>> ans(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  for (int i = 0; i < out.size(); ++i) {
    double *ptr = reinterpret_cast<double *>(taskData->outputs[i]);
    ans[i] = std::vector(ptr, ptr + matrixB.size());
  }
  std::vector<std::vector<double>> check_result = {{2, 0, 6}, {0, 14, 0}, {4, 0, 12}};
  ASSERT_EQ(check_result, ans);
}

// TEST(Sequential, Test_Sum_20) {
//   const int count = 20;
//
//   // Create data
//   std::vector<int> in(1, count);
//   std::vector<int> out(1, 0);
//
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   taskDataSeq->inputs_count.emplace_back(in.size());
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   nesterov_a_test_task_seq::TestTaskSequential testTaskSequential(taskDataSeq);
//   ASSERT_EQ(testTaskSequential.validation(), true);
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//   ASSERT_EQ(count, out[0]);
// }
//
// TEST(Sequential, Test_Sum_50) {
//   const int count = 50;
//
//   // Create data
//   std::vector<int> in(1, count);
//   std::vector<int> out(1, 0);
//
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   taskDataSeq->inputs_count.emplace_back(in.size());
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   nesterov_a_test_task_seq::TestTaskSequential testTaskSequential(taskDataSeq);
//   ASSERT_EQ(testTaskSequential.validation(), true);
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//   ASSERT_EQ(count, out[0]);
// }
//
// TEST(Sequential, Test_Sum_70) {
//   const int count = 70;
//
//   // Create data
//   std::vector<int> in(1, count);
//   std::vector<int> out(1, 0);
//
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   taskDataSeq->inputs_count.emplace_back(in.size());
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   nesterov_a_test_task_seq::TestTaskSequential testTaskSequential(taskDataSeq);
//   ASSERT_EQ(testTaskSequential.validation(), true);
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//   ASSERT_EQ(count, out[0]);
// }
//
// TEST(Sequential, Test_Sum_100) {
//   const int count = 100;
//
//   // Create data
//   std::vector<int> in(1, count);
//   std::vector<int> out(1, 0);
//
//   // Create TaskData
//   std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
//   taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
//   taskDataSeq->inputs_count.emplace_back(in.size());
//   taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
//   taskDataSeq->outputs_count.emplace_back(out.size());
//
//   // Create Task
//   nesterov_a_test_task_seq::TestTaskSequential testTaskSequential(taskDataSeq);
//   ASSERT_EQ(testTaskSequential.validation(), true);
//   testTaskSequential.pre_processing();
//   testTaskSequential.run();
//   testTaskSequential.post_processing();
//   ASSERT_EQ(count, out[0]);
// }
//
// int main(int argc, char **argv) {
//   testing::InitGoogleTest(&argc, argv);
//   return RUN_ALL_TESTS();
// }
