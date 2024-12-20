// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "seq/pikarychev_i_monte_carlo/include/ops_seq.hpp"

TEST(Sequential, MonteCarlo_Integration_Interval_0_10) {
  const double a = 0.0;
  const double b = 10.0;
  const int num_samples = 1000;
  const int seed = 42;
  double expected_result = 0.0;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(a, b);
  for (int i = 0; i < num_samples; i++) {
    double x = distribution(generator);
    expected_result += pikarychev_i_monte_carlo_seq::function_double(x);
  }
  expected_result *= (b - a) / num_samples;
  std::vector<double> in = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
  std::vector<double> out(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());
  pikarychev_i_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(expected_result, out[0], 1.0);
}

TEST(Sequential, MonteCarlo_Integration_Interval_0_50) {
  const double a = 0.0;
  const double b = 50.0;
  const int num_samples = 1000;
  const int seed = 42;
  double expected_result = 0.0;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(a, b);
  for (int i = 0; i < num_samples; i++) {
    double x = distribution(generator);
    expected_result += pikarychev_i_monte_carlo_seq::function_double(x);
  }
  expected_result *= (b - a) / num_samples;
  std::vector<double> in = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
  std::vector<double> out(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);
  pikarychev_i_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(expected_result, out[0], 1.0);
}

TEST(Sequential, MonteCarlo_Integration_Interval_0_100) {
  const double a = 0.0;
  const double b = 100.0;
  const int num_samples = 1000;
  const int seed = 42;
  double expected_result = 0.0;
  std::mt19937 generator(seed);
  std::uniform_real_distribution<double> distribution(a, b);
  for (int i = 0; i < num_samples; i++) {
    double x = distribution(generator);
    expected_result += pikarychev_i_monte_carlo_seq::function_double(x);
  }
  expected_result *= (b - a) / num_samples;
  std::vector<double> in = {a, b, static_cast<double>(num_samples), static_cast<double>(seed)};
  std::vector<double> out(1, 0.0);
  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(4);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(1);
  pikarychev_i_monte_carlo_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_TRUE(testTaskSequential.validation());
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(expected_result, out[0], 1.0);
}