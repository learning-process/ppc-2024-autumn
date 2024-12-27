// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <vector>

#include "seq/prokhorov_n_global_search_algorithm_strongin/include/ops_seq.hpp"

TEST(prokhorov_n_global_search_algorithm_strongin_seq, Test_Strongin_Algorithm_Quadratic_Function) {
  std::vector<double> in_a = {-10.0};
  std::vector<double> in_b = {10.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  taskDataSeq->inputs_count.emplace_back(in_a.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  taskDataSeq->inputs_count.emplace_back(in_b.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  taskDataSeq->inputs_count.emplace_back(in_epsilon.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_NEAR(out[0], 0.0, 0.001);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq, Test_Strongin_Algorithm_Cubic_Function) {
  std::vector<double> in_a = {-2.0};
  std::vector<double> in_b = {2.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  taskDataSeq->inputs_count.emplace_back(in_a.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  taskDataSeq->inputs_count.emplace_back(in_b.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  taskDataSeq->inputs_count.emplace_back(in_epsilon.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_NEAR(out[0], 0.0, 0.001);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq, Test_Strongin_Algorithm_Absolute_Function) {
  std::vector<double> in_a = {-5.0};
  std::vector<double> in_b = {5.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  taskDataSeq->inputs_count.emplace_back(in_a.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  taskDataSeq->inputs_count.emplace_back(in_b.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  taskDataSeq->inputs_count.emplace_back(in_epsilon.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_NEAR(out[0], 0.0, 0.001);
}

TEST(prokhorov_n_global_search_algorithm_strongin_seq, Test_Strongin_Algorithm_Exponential_Function) {
  std::vector<double> in_a = {-1.0};
  std::vector<double> in_b = {1.0};
  std::vector<double> in_epsilon = {0.001};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_a.data()));
  taskDataSeq->inputs_count.emplace_back(in_a.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_b.data()));
  taskDataSeq->inputs_count.emplace_back(in_b.size());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_epsilon.data()));
  taskDataSeq->inputs_count.emplace_back(in_epsilon.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  prokhorov_n_global_search_algorithm_strongin_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  EXPECT_NEAR(out[0], 0.0, 0.001);
}