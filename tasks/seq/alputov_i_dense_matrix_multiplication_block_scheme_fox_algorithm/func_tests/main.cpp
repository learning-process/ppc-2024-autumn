#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm/include/ops_seq.hpp"

namespace alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm {
std::vector<double> generator(int sz, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());

  if (a >= b) {
    throw std::invalid_argument("error.");
  }

  std::uniform_int_distribution<> dis(a, b);

  std::vector<double> ans(sz);
  for (int i = 0; i < sz; ++i) {
    ans[i] = dis(gen);
  }

  return ans;
}
}  // namespace alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, EmptyInput_ReturnsFalse1) {
  std::vector<double> A = {};
  std::vector<double> B = {};
  int column_A = 0;
  int row_A = 0;
  int column_B = 0;
  int row_B = 0;
  std::vector<double> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, EmptyInput_ReturnsFalse2) {
  std::vector<double> A = {};
  std::vector<double> B = {};
  int column_A = 1;
  int row_A = 0;
  int column_B = 0;
  int row_B = 0;
  std::vector<double> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, EmptyInput_ReturnsFalse3) {
  std::vector<double> A = {1.1};
  std::vector<double> B = {};
  int column_A = 1;
  int row_A = 1;
  int column_B = 0;
  int row_B = 0;
  std::vector<double> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, EmptyInput_ReturnsFalse4) {
  std::vector<double> A = {1.1};
  std::vector<double> B = {};
  int column_A = 1;
  int row_A = 1;
  int column_B = 1;
  int row_B = 0;
  std::vector<std::vector<double>> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, EmptyInput_ReturnsFalse5) {
  std::vector<double> A = {1.1};
  std::vector<double> B = {1.1, 2.2, 3.3, 4.4};
  int column_A = 1;
  int row_A = 1;
  int column_B = 2;
  int row_B = 2;
  std::vector<std::vector<double>> out(1);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, TestWithZerosA) {
  std::vector<double> A = {0.0, 0.0};
  std::vector<double> B = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(4, -1000, 1000);
  int column_A = 2;
  int row_A = 1;
  int column_B = 2;
  int row_B = 2;
  std::vector<double> out(2);
  std::vector<double> ans(2, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, TestWithZerosB) {
  std::vector<double> A = alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::generator(2, -1000, 1000);
  std::vector<double> B = {0.0, 0.0, 0.0, 0.0};
  int column_A = 2;
  int row_A = 1;
  int column_B = 2;
  int row_B = 2;
  std::vector<double> out(2);
  std::vector<double> ans(2, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, NonZeroElementsSimple) {
  std::vector<double> A = {1.0, 1.0};
  std::vector<double> B = {1.1, 1.1, 1.1, 1.1};
  int column_A = 2;
  int row_A = 1;
  int column_B = 2;
  int row_B = 2;
  std::vector<double> out(2);
  std::vector<double> ans(2, 2.2);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, NonZeroElementsSimple2) {
  std::vector<double> A = {1.0, 1.0, 1.0, 1.0};
  std::vector<double> B = {1.1, 1.1, 1.1, 1.1};
  int column_A = 2;
  int row_A = 2;
  int column_B = 2;
  int row_B = 2;
  std::vector<double> out(4);
  std::vector<double> ans(4, 2.2);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans, out);
}

TEST(alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm_seq, HardTest) {
  std::vector<double> A = {1.1, 2.2, 3.3, 4.4};
  std::vector<double> B = {5.5, 6.6, 7.7, 8.8};
  int column_A = 2;
  int row_A = 2;
  int column_B = 2;
  int row_B = 2;
  std::vector<double> out(4);
  std::vector<double> ans = {22.99, 26.62, 52.03, 60.5};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs_count.emplace_back(row_A);
  taskDataSeq->inputs_count.emplace_back(column_A);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(row_B);
  taskDataSeq->inputs_count.emplace_back(column_B);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  alputov_i_dense_matrix_multiplication_block_scheme_fox_algorithm::
      dense_matrix_multiplication_block_scheme_fox_algorithm_seq testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (int i = 0; i < row_A * column_B; ++i) {
    ASSERT_NEAR(ans[i], out[i], 1e-5);
  }
}