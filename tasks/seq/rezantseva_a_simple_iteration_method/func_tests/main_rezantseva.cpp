// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include "seq/rezantseva_a_simple_iteration_method/include/ops_seq_rezantseva.hpp"

TEST(rezantseva_a_simple_iteration_method_seq, check_matrix_3x3) {
  size_t size = 3;
  const double eps = 1e-3;
  // Create data
  std::vector<double> A = {4.0, 1.0, 2.0, 5.0, 9.0, 3.0, 4.0, 2.0, 12.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> answer = {0.23, 0.0971, -0.00944};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < size; ++i) {
    ASSERT_NEAR(out[i], answer[i], eps);
  }
}

TEST(rezantseva_a_simple_iteration_method_seq, check_matrix_4x4) {
  size_t size = 4;
  const double eps = 1e-3;
  // Create data
  std::vector<double> A = {5.0, 1.0, 1.0, 1.0, 1.0, 7.0, 1.0, 1.0, 1.0, 1.0, 6.0, 1.0, 1.0, 1.0, 1.0, 4.0};
  std::vector<double> b = {8.0, 10.0, 9.0, 7.0};
  std::vector<double> answer = {1.0, 1.0, 1.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < size; ++i) {
    ASSERT_NEAR(out[i], answer[i], eps);
  }
}

TEST(rezantseva_a_simple_iteration_method_seq, check_matrix_5x5) {
  size_t size = 5;
  const double eps = 1e-3;
  // Create data
  std::vector<double> A = {10.0, -2.0, -1.0, -1.0, -1.0, -2.0, 12.0, -3.0, -1.0, -1.0, -1.0, -3.0, 15.0,
                           -1.0, -1.0, -1.0, -1.0, -1.0, 11.0, -2.0, -1.0, -1.0, -1.0, -2.0, 14.0};
  std::vector<double> b = {8.0, 10.0, 9.0, 7.0, 11.0};
  std::vector<double> answer = {1.488, 1.587, 1.185, 1.254, 1.269};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < size; ++i) {
    ASSERT_NEAR(out[i], answer[i], eps);
  }
}

TEST(rezantseva_a_simple_iteration_method_seq, check_matrix_6x6) {
  size_t size = 6;
  const double eps = 1e-3;
  // Create data
  std::vector<double> A = {10.0, -2.0, -1.0, -1.0, -1.0, -3.0, -2.0, 12.0, -3.0, -1.0, -1.0, -3.0,
                           -1.0, -3.0, 15.0, -1.0, -1.0, -2.0, -1.0, -1.0, -1.0, 11.0, -2.0, -3.0,
                           -1.0, -1.0, -1.0, -2.0, 14.0, -5.0, -1.0, -2.0, -1.0, -1.0, -3.0, 17};
  std::vector<double> b = {8.0, 10.0, 9.0, 7.0, 11.0, 5.0};
  std::vector<double> answer = {2.144, 2.209, 1.598, 1.857, 1.912, 1.22};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  for (size_t i = 0; i < size; ++i) {
    ASSERT_NEAR(out[i], answer[i], eps);
  }
}

TEST(rezantseva_a_simple_iteration_method_seq, check_validation_invalid_inputs_count) {
  size_t size = 3;
  // Create data
  std::vector<double> A = {4.0, 1.0, 2.0, 5.0, 9.0, 3.0, 4.0, 2.0, 12.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(rezantseva_a_simple_iteration_method_seq, check_validation_invalid_outputs_count) {
  size_t size = 3;
  // Create data
  std::vector<double> A = {4.0, 1.0, 2.0, 5.0, 9.0, 3.0, 4.0, 2.0, 12.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));

  // Create Task
  rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(rezantseva_a_simple_iteration_method_seq, check_validation_invalid_matrix_size) {
  size_t size = 0;
  // Create data
  std::vector<double> A = {4.0, 1.0, 2.0, 5.0, 9.0, 3.0, 4.0, 2.0, 12.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(rezantseva_a_simple_iteration_method_seq, check_validation_invalid_matrix) {
  size_t size = 3;
  // Create data
  std::vector<double> A = {4.0, 1.0, 7.0, 5.0, 7.0, 3.0, 4.0, 2.0, 5.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(rezantseva_a_simple_iteration_method_seq, check_validation_zero_matrix) {
  size_t size = 3;
  // Create data
  std::vector<double> A = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(rezantseva_a_simple_iteration_method_seq, check_validation_zero_diagonal_matrix) {
  size_t size = 3;
  // Create data
  std::vector<double> A = {0.0, 2.0, -2.0, 5.0, 0.0, 6.0, 7.0, -1.0, 0.0};
  std::vector<double> b = {1.0, 2.0, 1.0};
  std::vector<double> out(size, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&size));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(A.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(b.data()));

  taskDataSeq->inputs_count.emplace_back(size);
  taskDataSeq->inputs_count.emplace_back(A.size());
  taskDataSeq->inputs_count.emplace_back(b.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_simple_iteration_method_seq::SimpleIterationSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}