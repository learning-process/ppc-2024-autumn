#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "seq/zolotareva_a_SLE_gradient_method/include/ops_seq.hpp"
namespace zolotareva_a_SLE_gradient_method_seq {
void generateSLE(std::vector<double>& A, std::vector<double>& b, int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-10.0f, 10.0f);

  for (int i = 0; i < n; ++i) {
    b[i] = dist(gen);
    for (int j = i; j < n; ++j) {
      double value = dist(gen);
      A[i * n + j] = value;
      A[j * n + i] = value;
    }
  }

  for (int i = 0; i < n; ++i) {
    A[i * n + i] += n * 10.0f;
  }
}

void form(int n_) {
  int n = n_;
  std::vector<double> A(n * n);
  std::vector<double> b(n);
  std::vector<double> x(n);
  generateSLE(A, b, n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(n);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(x.size());

  zolotareva_a_SLE_gradient_method_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
      sum += A[i * n + j] * x[j];
    }
    EXPECT_NEAR(sum, b[i], 1e-5);
  }
}
}  // namespace zolotareva_a_SLE_gradient_method_seq

TEST(zolotareva_a_SLE_gradient_method_seq, invalid_input_sizes) {
  int n = 2;
  std::vector<double> A = {2, -1, -1, 2};
  std::vector<double> b = {1, 3, 4};  // Неправильный размер b
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(b.size());
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(n);

  zolotareva_a_SLE_gradient_method_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_FALSE(task.validation());
}

TEST(zolotareva_a_SLE_gradient_method_seq, non_symmetric_matrix) {
  int n = 2;
  std::vector<double> A = {2, -1, 0, 2};  // A[0][1] != A[1][0]
  std::vector<double> b = {1, 3};
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(n);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(n);

  zolotareva_a_SLE_gradient_method_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), false);
}

TEST(zolotareva_a_SLE_gradient_method_seq, not_positive_definite_matrix) {
  int n = 2;
  std::vector<double> A = {0, 0, 0, 0};
  std::vector<double> b = {0, 0};
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(n);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(n);

  zolotareva_a_SLE_gradient_method_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), false);
}

TEST(zolotareva_a_SLE_gradient_method_seq, negative_definite_matrix) {
  int n = 2;
  std::vector<double> A = {-1, 0, 0, -2};
  std::vector<double> b = {1, 1};
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(n);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(n);

  zolotareva_a_SLE_gradient_method_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), false);
}

TEST(zolotareva_a_SLE_gradient_method_seq, zero_dimension) {
  std::vector<double> A;
  std::vector<double> b;
  std::vector<double> x;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs_count.push_back(0);
  taskDataSeq->inputs_count.push_back(0);
  taskDataSeq->outputs_count.push_back(0);

  zolotareva_a_SLE_gradient_method_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), false);
}

TEST(zolotareva_a_SLE_gradient_method_seq, singular_matrix) {
  int n = 2;
  std::vector<double> A = {1, 1, 1, 1};  // Сингулярная матрица
  std::vector<double> b = {2, 2};
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(n);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(n);

  zolotareva_a_SLE_gradient_method_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), false);
}

TEST(zolotareva_a_SLE_gradient_method_seq, zero_vector_solution) {
  int n = 2;
  std::vector<double> A = {1, 0, 0, 1};
  std::vector<double> b = {0, 0};
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(n);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(n);

  zolotareva_a_SLE_gradient_method_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], 0.0, 1e-2);  // Ожидаем нулевой вектор решения
  }
}

TEST(zolotareva_a_SLE_gradient_method_seq, n_equals_one) {
  int n = 1;
  std::vector<double> A = {2};
  std::vector<double> b = {4};
  std::vector<double> x(n);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(n);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(n);

  zolotareva_a_SLE_gradient_method_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();

  EXPECT_NEAR(x[0], 2.0, 1e-1);  // Ожидаемое решение x = 2
}

TEST(zolotareva_a_SLE_gradient_method_seq, test_correct_answer1) {
  int n = 3;
  std::vector<double> A = {4, -1, 2, -1, 6, -2, 2, -2, 5};
  std::vector<double> b = {-1, 9, -10};
  std::vector<double> x;
  x.resize(n);
  std::vector<double> ref_x = {1, 1, -2};

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(A.data()));
  taskDataSeq->inputs.push_back(reinterpret_cast<uint8_t*>(b.data()));
  taskDataSeq->inputs_count.push_back(n * n);
  taskDataSeq->inputs_count.push_back(n);
  taskDataSeq->outputs.push_back(reinterpret_cast<uint8_t*>(x.data()));
  taskDataSeq->outputs_count.push_back(x.size());

  zolotareva_a_SLE_gradient_method_seq::TestTaskSequential task(taskDataSeq);
  ASSERT_EQ(task.validation(), true);
  task.pre_processing();
  task.run();
  task.post_processing();
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(x[i], ref_x[i], 1e-12);
  }
}
TEST(zolotareva_a_SLE_gradient_method_seq, Test_Image_random_n_3) { zolotareva_a_SLE_gradient_method_seq::form(3); };
TEST(zolotareva_a_SLE_gradient_method_seq, Test_Image_random_n_5) { zolotareva_a_SLE_gradient_method_seq::form(5); };
TEST(zolotareva_a_SLE_gradient_method_seq, Test_Image_random_n_7) { zolotareva_a_SLE_gradient_method_seq::form(7); };
TEST(zolotareva_a_SLE_gradient_method_seq, Test_Image_random_n_20) { zolotareva_a_SLE_gradient_method_seq::form(591); };
