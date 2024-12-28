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
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  morozov_e_mult_sparse_matrix::fillData(taskData, matrixA.size(), matrixA[0].size(), matrixB.size(), matrixB[0].size(),
                                         dA, row_indA, col_indA, dB, row_indB, col_indB, out);

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
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  morozov_e_mult_sparse_matrix::fillData(taskData, matrixA.size(), matrixA[0].size(), matrixB.size(), matrixB[0].size(),
                                         dA, row_indA, col_indA, dB, row_indB, col_indB, out);

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
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size() + 1, 0));
  morozov_e_mult_sparse_matrix::fillData(taskData, matrixA.size(), matrixA[0].size(), matrixB.size(), matrixB[0].size(),
                                         dA, row_indA, col_indA, dB, row_indB, col_indB, out);

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
  std::vector<std::vector<double>> out(matrixA.size() + 1, std::vector<double>(matrixB[0].size(), 0));
  morozov_e_mult_sparse_matrix::fillData(taskData, matrixA.size(), matrixA[0].size(), matrixB.size(), matrixB[0].size(),
                                         dA, row_indA, col_indA, dB, row_indB, col_indB, out);

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
  std::vector<std::vector<double>> out(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  morozov_e_mult_sparse_matrix::fillData(taskData, matrixA.size(), matrixA[0].size(), matrixB.size(), matrixB[0].size(),
                                         dA, row_indA, col_indA, dB, row_indB, col_indB, out);
  morozov_e_mult_sparse_matrix::TestTaskSequential testTaskSequential(taskData);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  std::vector<std::vector<double>> ans(matrixA.size(), std::vector<double>(matrixB[0].size(), 0));
  for (size_t i = 0; i < out.size(); ++i) {
    auto *ptr = reinterpret_cast<double *>(taskData->outputs[i]);
    ans[i] = std::vector(ptr, ptr + matrixB.size());
  }
  std::vector<std::vector<double>> check_result = {{2, 0, 6}, {0, 14, 0}, {4, 0, 12}};
  ASSERT_EQ(check_result, ans);
}