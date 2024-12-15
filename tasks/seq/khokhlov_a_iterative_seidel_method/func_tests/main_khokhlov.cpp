#include <gtest/gtest.h>

#include "seq/khokhlov_a_iterative_seidel_method/include/ops_seq_khokhlov.hpp"

void getRandomSLAU(std::vector<double> &A, std::vector<double> &b, int N) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < N; ++i) {
    double rowSum = 0.0;
    for (int j = 0; j < N; ++j) {
      if (i != j) {
        A[i * N + j] = rand() % 10 - 5;
        rowSum += std::abs(A[i * N + j]);
      }
    }
    A[i * N + i] = rowSum + (rand() % 5 + 1);
    b[i] = rand() % 20 - 10;
  }
}

TEST(khokhlov_a_iterative_seidel_method_seq, test_empty_matrix) {
  const int n = 0;
  const int maxiter = 0;
  const double eps = 1e-6;

  // create data
  std::vector<double> A = {};
  std::vector<double> b = {};
  std::vector<double> expect = {};
  std::vector<double> result;

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskdataSeq->inputs_count.emplace_back(n);
  taskdataSeq->inputs_count.emplace_back(maxiter);
  taskdataSeq->inputs_count.emplace_back(eps);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskdataSeq->outputs_count.emplace_back(result.size());

  // crate task
  khokhlov_a_iterative_seidel_method_seq::seidel_method_seq seidel_method_seq(taskdataSeq);
  ASSERT_FALSE(seidel_method_seq.validation());
}

TEST(khokhlov_a_iterative_seidel_method_seq, test_matrix_with_invalid_iter) {
  const int n = 3;
  const int maxiter = 0;
  const double eps = 1e-6;

  // create data
  std::vector<double> A = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::vector<double> b = {1, 2, 3};
  std::vector<double> expect = {};
  std::vector<double> result;

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskdataSeq->inputs_count.emplace_back(n);
  taskdataSeq->inputs_count.emplace_back(maxiter);
  taskdataSeq->inputs_count.emplace_back(eps);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskdataSeq->outputs_count.emplace_back(result.size());

  // crate task
  khokhlov_a_iterative_seidel_method_seq::seidel_method_seq seidel_method_seq(taskdataSeq);
  ASSERT_FALSE(seidel_method_seq.validation());
}

TEST(khokhlov_a_iterative_seidel_method_seq, test_3x3_matrix_with_1000_iter) {
  const int n = 3;
  const int maxiter = 1000;
  const double eps = 1e-6;

  // create data
  std::vector<double> A = {4.0, -1.0, 1.0, 2.0, 5.0, 2.0, 1.0, -1.0, 6.0};
  std::vector<double> b = {5.0, 7.0, 8.0};
  std::vector<double> expect = {1.0, 0.5, 1.25};
  std::vector<double> result(n, 0.0);

  // create task data
  std::shared_ptr<ppc::core::TaskData> taskdataSeq = std::make_shared<ppc::core::TaskData>();
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskdataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));
  taskdataSeq->inputs_count.emplace_back(n);
  taskdataSeq->inputs_count.emplace_back(maxiter);
  taskdataSeq->inputs_count.emplace_back(eps);
  taskdataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskdataSeq->outputs_count.emplace_back(result.size());

  // crate task
  khokhlov_a_iterative_seidel_method_seq::seidel_method_seq seidel_method_seq(taskdataSeq);
  ASSERT_TRUE(seidel_method_seq.validation());
  seidel_method_seq.pre_processing();
  seidel_method_seq.run();
  seidel_method_seq.post_processing();

  for (int i = 0; i < n; i++) {
    ASSERT_NEAR(expect[i], result[i], 1e-1);
  }
}
