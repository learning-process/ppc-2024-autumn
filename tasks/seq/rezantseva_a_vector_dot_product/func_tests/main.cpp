// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include "seq/rezantseva_a_vector_dot_product/include/ops_seq.hpp"

TEST(rezantseva_a_vector_dot_product_seq, can_scalar_multiply_vec_size_10) {
  const int count = 10;
  // Create data
  std::vector<int> out(1, 0);
  std::vector<int> v1 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs_count.emplace_back(v2.size());
  
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  int answer = rezantseva_a_vector_dot_product_seq::vectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}

TEST(rezantseva_a_vector_dot_product_seq, can_scalar_multiply_vec_size_100) {
  const int count = 100;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs_count.emplace_back(v2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  int answer = rezantseva_a_vector_dot_product_seq::vectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}

TEST(rezantseva_a_vector_dot_product_seq, check_none_equal_size_of_vec) {
  const int count = 10;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_seq::createRandomVector(count + 1);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs_count.emplace_back(v2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(rezantseva_a_vector_dot_product_seq, check_equal_size_of_vec) {
  const int count = 10;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs_count.emplace_back(v2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
}

TEST(rezantseva_a_vector_dot_product_seq, check_empty_vec_product_func) {
  const int count = 0;
  std::vector<int> v1 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  int answer = rezantseva_a_vector_dot_product_seq::vectorDotProduct(v1, v2);
  ASSERT_EQ(0, answer);
}

TEST(rezantseva_a_vector_dot_product_seq, check_empty_vec_product_run) {
  const int count = 0;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs_count.emplace_back(v2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  int answer = rezantseva_a_vector_dot_product_seq::vectorDotProduct(v1, v2);
  ASSERT_EQ(answer, out[0]);
}

TEST(rezantseva_a_vector_dot_product_seq, v1_dot_product_v2_equal_v2_dot_product_v1) {
  const int count = 50;
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);
  std::vector<int> v2 = rezantseva_a_vector_dot_product_seq::createRandomVector(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs_count.emplace_back(v2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  int answer = rezantseva_a_vector_dot_product_seq::vectorDotProduct(v2, v1);
  ASSERT_EQ(answer, out[0]);
}
TEST(rezantseva_a_vector_dot_product_seq, check_run_right) {
  // Create data
  std::vector<int> out(1, 0);

  std::vector<int> v1 = {1, 2, 5};
  std::vector<int> v2 = {4, 7, 8};

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v2.data()));

  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->inputs_count.emplace_back(v2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_a_vector_dot_product_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  
  ASSERT_EQ(58, out[0]);
}
TEST(rezantseva_a_vector_dot_product_seq, check_vectorDotProduct_right) {
  // Create data
  std::vector<int> v1 = {1, 2, 5};
  std::vector<int> v2 = {4, 7, 8};
  ASSERT_EQ(58, rezantseva_a_vector_dot_product_seq::vectorDotProduct(v1, v2));
}

TEST(rezantseva_a_vector_dot_product_seq, can_throw_exp_vectorDotProduct) {
  // Create data
  std::vector<int> v1 = {1, 2, 5, 7};
  std::vector<int> v2 = {4, 7, 8};
  ASSERT_ANY_THROW(rezantseva_a_vector_dot_product_seq::vectorDotProduct(v1, v2));
}
