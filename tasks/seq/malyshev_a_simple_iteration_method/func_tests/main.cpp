// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/malyshev_a_simple_iteration_method/include/matrix.hpp"
#include "seq/malyshev_a_simple_iteration_method/include/ops_seq.hpp"

TEST(malyshev_a_simple_iteration_method_seq, basic_test) {
  // Create data
  std::vector<double> A{3, 2, -1, 1, -2, 1, 2, -3, -5};
  std::vector<double> B{8, -2, 1};
  std::vector<double> X(3, 0);
  std::vector<double> X0(3, 0);
  double eps = 1e-4;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(X.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->outputs_count.emplace_back(X.size());

  // Create Task
  malyshev_a_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double sum_eq;
  for (uint32_t i = 0; i < X.size(); i++) {
    sum_eq = 0;
    for (uint32_t j = 0; j < X.size(); j++) {
      sum_eq += X[j] * A[i * X.size() + j];
    }
    ASSERT_TRUE(std::abs(sum_eq - B[i]) <= eps);
  }
}

TEST(malyshev_a_simple_iteration_method_seq, random_test) {
  // Create data
  const int size = 10;
  std::vector<double> A;
  std::vector<double> B;
  malyshev_a_simple_iteration_method_seq::getRandomData(size, A, B);

  std::vector<double> X(size, 0);
  std::vector<double> X0(size, 0);
  double eps = 1e-4;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(X.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->outputs_count.emplace_back(X.size());

  // Create Task
  malyshev_a_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  double sum_eq;
  for (uint32_t i = 0; i < X.size(); i++) {
    sum_eq = 0;
    for (uint32_t j = 0; j < X.size(); j++) {
      sum_eq += X[j] * A[i * X.size() + j];
    }
    ASSERT_TRUE(std::abs(sum_eq - B[i]) <= eps);
  }
}

TEST(malyshev_a_simple_iteration_method_seq, validate_input_data) {
  std::vector<double> A;
  std::vector<double> B;
  std::vector<double> X;
  std::vector<double> X0;
  double eps = 1e-4;

  const auto try_validate = [](auto &taskData) {
    malyshev_a_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskData);
    return testTaskSequential.validation();
  };

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  ASSERT_FALSE(try_validate(taskDataSeq));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  ASSERT_FALSE(try_validate(taskDataSeq));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  ASSERT_FALSE(try_validate(taskDataSeq));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  ASSERT_FALSE(try_validate(taskDataSeq));
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  ASSERT_FALSE(try_validate(taskDataSeq));
  taskDataSeq->inputs_count.emplace_back(X.size());
  ASSERT_FALSE(try_validate(taskDataSeq));
}

TEST(malyshev_a_simple_iteration_method_seq, validate_input_determinant) {
  std::vector<double> A{3, 0, -1, 1, 0, 1, 2, 0, -5};
  std::vector<double> X(3, 0);
  std::vector<double> B{8, -2, 1};
  std::vector<double> X0(3, 0);
  double eps = 1e-4;

  const auto try_validate = [](auto &taskData) {
    malyshev_a_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskData);
    return testTaskSequential.validation();
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(X.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->outputs_count.emplace_back(X.size());

  ASSERT_FALSE(try_validate(taskDataSeq));
}

TEST(malyshev_a_simple_iteration_method_seq, validate_input_rank) {
  std::vector<double> A{1, 1, 3, 3};
  std::vector<double> X(2, 0);
  std::vector<double> B{1, 2};
  std::vector<double> X0(2, 0);
  double eps = 1e-4;

  const auto try_validate = [](auto &taskData) {
    malyshev_a_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskData);
    return testTaskSequential.validation();
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(X.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->outputs_count.emplace_back(X.size());

  ASSERT_FALSE(try_validate(taskDataSeq));
}

TEST(malyshev_a_simple_iteration_method_seq, validate_input_slowly_converging) {
  std::vector<double> A{5, -7, 3, 2};
  std::vector<double> X(2, 0);
  std::vector<double> B{1, 2};
  std::vector<double> X0(2, 0);
  double eps = 1e-4;

  const auto try_validate = [](auto &taskData) {
    malyshev_a_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskData);
    return testTaskSequential.validation();
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(X.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->outputs_count.emplace_back(X.size());

  ASSERT_FALSE(try_validate(taskDataSeq));
}

TEST(malyshev_a_simple_iteration_method_seq, validate_input_zero_on_the_main_diagonal) {
  std::vector<double> A{4, 2, -3, 0};
  std::vector<double> X(2, 0);
  std::vector<double> B{1, 2};
  std::vector<double> X0(2, 0);
  double eps = 1e-4;

  const auto try_validate = [](auto &taskData) {
    malyshev_a_simple_iteration_method_seq::TestTaskSequential testTaskSequential(taskData);
    return testTaskSequential.validation();
  };

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(B.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(X0.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
  taskDataSeq->inputs_count.emplace_back(X.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(X.data()));
  taskDataSeq->outputs_count.emplace_back(X.size());

  ASSERT_TRUE(try_validate(taskDataSeq));
}