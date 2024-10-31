// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <random>

#include "seq/koshkin_m_scalar_product_of_vectors/include/ops_seq.hpp"
static int offset = 0;

std::vector<int> createRandomVector(int v_size) {
  std::vector<int> vec(v_size);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);
  for (int i = 0; i < v_size; i++) vec[i] = gen() % 100;
  return vec;
}

TEST(koshkin_m_scalar_product_of_vectors, Test1) {
  const int count = 0;
  std::vector<int> vec_1 = createRandomVector(count);
  std::vector<int> vec_2 = createRandomVector(count);
  int ans = koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2);
  ASSERT_EQ(0, ans);
}

TEST(koshkin_m_scalar_product_of_vectors, can_scalar_multiply_vec_size_10) {
  const int count = 10;
  // Create data
  std::vector<int> out(1, 0);
  std::vector<int> vec_1 = createRandomVector(count);
  std::vector<int> vec_2 = createRandomVector(count);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_1.data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec_2.data()));

  taskDataSeq->inputs_count.emplace_back(vec_1.size());
  taskDataSeq->inputs_count.emplace_back(vec_2.size());

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  koshkin_m_scalar_product_of_vectors::VectorDotProduct VectorDotProduct(taskDataSeq);
  ASSERT_EQ(VectorDotProduct.validation(), true);
  VectorDotProduct.pre_processing();
  VectorDotProduct.run();
  VectorDotProduct.post_processing();
  int answer = koshkin_m_scalar_product_of_vectors::calculateDotProduct(vec_1, vec_2);
  ASSERT_EQ(answer, out[0]);
}

TEST(koshkin_m_scalar_product_of_vectors, check_calculateDotProduct_right) {
  // Create data
  std::vector<int> v1 = {1, 2, 5};
  std::vector<int> v2 = {4, 7, 8};
  ASSERT_EQ(58, koshkin_m_scalar_product_of_vectors::calculateDotProduct(v1, v2));
}
