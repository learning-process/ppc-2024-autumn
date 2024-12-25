// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/drozhdinov_d_mult_matrix_fox/include/ops_seq.hpp"
using namespace drozhdinov_d_mult_matrix_fox_seq;
namespace drozhdinov_d_mult_matrix_fox_seq {
std::vector<double> MatrixMult(const std::vector<double> &A, const std::vector<double> &B, int k, int l, int n) {
  std::vector<double> result(k * n, 0.0);

  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      for (int p = 0; p < l; p++) {
        result[i * n + j] += A[i * l + p] * B[p * n + j];
      }
    }
  }

  return result;
}

std::vector<double> getRandomMatrix(int sz, int lbound, int rbound) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<double> vec(sz);
  std::uniform_int_distribution<int> dist(lbound, rbound);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}
}  // namespace drozhdinov_d_mult_matrix_fox_seq

TEST(drozhdinov_d_mult_matrix_fox_seq, 2x3_3x2size) {
  int k = 2;
  int l = 3;
  int m = 3;
  int n = 2;
  std::vector<double> A = {1, 2, 3, 4, 5, 6};
  std::vector<double> B = {7, 8, 9, 10, 11, 12};
  std::vector<double> res(4);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (int i = 0; i < k * n; i++) {
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
  }
}

TEST(drozhdinov_d_mult_matrix_fox_seq, Random50x200_200x50Test) {
  int k = 50;
  int l = 100;
  int m = 100;
  int n = 50;
  std::vector<double> A = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> res(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (int i = 0; i < k * n; i++) {
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
  }
}

TEST(drozhdinov_d_mult_matrix_fox_seq, Random100x100Test) {
  int k = 100;
  int l = 100;
  int m = 100;
  int n = 100;
  std::vector<double> A = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> res(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (int i = 0; i < k * n; i++) {
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
  }
}

TEST(drozhdinov_d_mult_matrix_fox_seq, EmptyTest) {
  int k = 0;
  int l = 0;
  int m = 0;
  int n = 0;
  std::vector<double> A = {};
  std::vector<double> B = {};
  std::vector<double> res;
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(expres, res);
}

TEST(drozhdinov_d_mult_matrix_fox_seq, 1x10_10x7Random) {
  int k = 1;
  int l = 10;
  int m = 10;
  int n = 7;
  std::vector<double> A = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(m * n, -100, 100);
  std::vector<double> res(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (int i = 0; i < k * n; i++) {
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
  }
}

TEST(drozhdinov_d_mult_matrix_fox_seq, 1x1_1x1Random) {
  int k = 1;
  int l = 1;
  int m = 1;
  int n = 1;
  std::vector<double> A = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(m * n, -100, 100);
  std::vector<double> res(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  for (int i = 0; i < k * n; i++) {
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
    EXPECT_DOUBLE_EQ(res[i], expres[i]);
  }
}

TEST(drozhdinov_d_mult_matrix_fox_seq, WrongValidation1) {
  int k = 1;
  int l = 1;
  int m = 2;
  int n = 1;  // A cols not equal B rows
  std::vector<double> A = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(m * n, -100, 100);
  std::vector<double> res(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(drozhdinov_d_mult_matrix_fox_seq, WrongValidation2) {
  int k = 1;
  int l = 1;
  int m = 1;
  int n = 1;
  std::vector<double> A = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(m * n, -100, 100);
  std::vector<double> res(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));  // only 1 matrix given
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(drozhdinov_d_mult_matrix_fox_seq, WrongValidation3) {
  int k = 1;
  int l = 1;
  int m = 1;
  int n = 1;
  std::vector<double> A = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(m * n, -100, 100);
  std::vector<double> res(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);  // outputs data lost
  taskDataSeq->outputs_count.emplace_back(k);
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(drozhdinov_d_mult_matrix_fox_seq, WrongValidation4) {
  int k = 1;
  int l = 1;
  int m = 1;
  int n = 1;
  std::vector<double> A = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(k * l, -100, 100);
  std::vector<double> B = drozhdinov_d_mult_matrix_fox_seq::getRandomMatrix(m * n, -100, 100);
  std::vector<double> res(k * n);
  std::vector<double> expres = MatrixMult(A, B, k, l, n);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs_count.emplace_back(k);
  taskDataSeq->inputs_count.emplace_back(l);
  taskDataSeq->inputs_count.emplace_back(m);
  taskDataSeq->inputs_count.emplace_back(n);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res.data()));
  taskDataSeq->outputs_count.emplace_back(k + 1);  // outputs_count wrong
  taskDataSeq->outputs_count.emplace_back(n);

  // Create Task
  drozhdinov_d_mult_matrix_fox_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}