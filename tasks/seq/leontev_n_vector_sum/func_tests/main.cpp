// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/leontev_n_vector_sum/include/ops_seq.hpp"

template <class InOutType>
void taskEmplacement(std::shared_ptr<ppc::core::TaskData> &taskDataPar, std::vector<InOutType> &global_vec,
                     std::vector<InOutType> &global_sum) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
  taskDataPar->outputs_count.emplace_back(global_sum.size());
}

TEST(leontev_n_vector_sum_seq, int_vector_sum) {
  // Create data
  std::vector<int32_t> in(5, 10);
  const int32_t expected_sum = 50;
  std::vector<int32_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<int32_t>(taskDataSeq, in, out);

  // Create Task
  leontev_n_vector_sum_seq::VecSumSequential<int32_t> vecSumSequential(taskDataSeq);
  ASSERT_TRUE(vecSumSequential.validation());
  vecSumSequential.pre_processing();
  vecSumSequential.run();
  vecSumSequential.post_processing();
  ASSERT_EQ(expected_sum, out[0]);
}

TEST(leontev_n_vector_sum_seq, double_vector_sum) {
  // Create data
  std::vector<double> in(5, 10.0);
  const double expected_sum = 50.0;
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<double>(taskDataSeq, in, out);

  // Create Task
  leontev_n_vector_sum_seq::VecSumSequential<double> vecSumSequential(taskDataSeq);
  ASSERT_TRUE(vecSumSequential.validation());
  vecSumSequential.pre_processing();
  vecSumSequential.run();
  vecSumSequential.post_processing();
  EXPECT_NEAR(out[0], expected_sum, 1e-6);
}

TEST(leontev_n_vector_sum_seq, float_vector_sum) {
  // Create data
  std::vector<float> in(5, 1.f);
  std::vector<float> out(1, 0.f);
  const float expected_sum = 5.f;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<float>(taskDataSeq, in, out);

  // Create Task
  leontev_n_vector_sum_seq::VecSumSequential<float> vecSumSequential(taskDataSeq);
  ASSERT_TRUE(vecSumSequential.validation());
  vecSumSequential.pre_processing();
  vecSumSequential.run();
  vecSumSequential.post_processing();
  EXPECT_NEAR(out[0], expected_sum, 1e-3f);
}

TEST(leontev_n_vector_sum_seq, int32_vector_sum) {
  // Create data
  std::vector<int32_t> in(2000, 5);
  std::vector<int32_t> out(1, 0);
  const int32_t expected_sum = 10000;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<int32_t>(taskDataSeq, in, out);

  // Create Task
  leontev_n_vector_sum_seq::VecSumSequential<int32_t> vecSumSequential(taskDataSeq);
  ASSERT_TRUE(vecSumSequential.validation());
  vecSumSequential.pre_processing();
  vecSumSequential.run();
  vecSumSequential.post_processing();
  ASSERT_EQ(out[0], expected_sum);
}

TEST(leontev_n_vector_sum_seq, uint32_vector_sum) {
  // Create data
  std::vector<uint32_t> in(255, 2);
  std::vector<uint32_t> out(1, 0);
  const uint32_t expected_sum = 510;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<uint32_t>(taskDataSeq, in, out);

  // Create Task
  leontev_n_vector_sum_seq::VecSumSequential<uint32_t> vecSumSequential(taskDataSeq);
  ASSERT_TRUE(vecSumSequential.validation());
  vecSumSequential.pre_processing();
  vecSumSequential.run();
  vecSumSequential.post_processing();
  ASSERT_EQ(out[0], expected_sum);
}

TEST(leontev_n_vector_sum_seq, empty_array_vector_sum) {
  // Create data
  std::vector<int32_t> in(1, 0);
  const int32_t expected_sum = 0;
  std::vector<int32_t> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskEmplacement<int32_t>(taskDataSeq, in, out);

  // Create Task
  leontev_n_vector_sum_seq::VecSumSequential<int32_t> vecSumSequential(taskDataSeq);
  ASSERT_TRUE(vecSumSequential.validation());
  vecSumSequential.pre_processing();
  vecSumSequential.run();
  vecSumSequential.post_processing();
  ASSERT_EQ(expected_sum, out[0]);
}
