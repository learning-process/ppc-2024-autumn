// Copyright 2024 Ivanov Mike
#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <numbers>
#include <vector>

#include "seq/ivanov_m_integration_trapezoid/include/ops_seq.hpp"

TEST(ivanov_m_integration_trapezoid_seq_func_test, simple_parabola) {
  const double a = 0;
  const double b = 1;
  const int n = 1000;
  const double res = 1.0 / 3.0;

  // Create function y = x^2
  std::function<double(double)> _f = [](double x) { return x * x; };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_seq_func_test, simple_parabola_swapped_borders) {
  const double a = 1;
  const double b = 0;
  const int n = 1000;
  const double res = -1.0 / 3.0;

  // Create function y = x^2
  std::function<double(double)> _f = [](double x) { return x * x; };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_seq_func_test, line_function) {
  const double a = 0;
  const double b = 5;
  const int n = 1000;
  const double res = 15.0;

  // Create function y = 2*(x-1)
  std::function<double(double)> _f = [](double x) { return 2 * (x - 1); };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_seq_func_test, sinus) {
  const double a = 0;
  const auto b = static_cast<double>(std::numbers::pi);
  const int n = 1000;
  const double res = 2.0;

  // Create function y = sin(x)
  std::function<double(double)> _f = [](double x) { return sin(x); };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_seq_func_test, sqrt) {
  const double a = 0;
  const double b = 4;
  const int n = 1000;
  const double res = 16.0 / 3.0;

  // Create function
  std::function<double(double)> _f = [](double x) { return sqrt(x); };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_seq_func_test, simple_ln) {
  const double a = 1;
  const auto b = static_cast<double>(std::numbers::e);
  const int n = 1000;
  const double res = 1.0;

  // Create function
  std::function<double(double)> _f = [](double x) { return log(x); };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_seq_func_test, ln_with_sqr_border_exp) {
  const double a = 1;
  const auto b = static_cast<double>(std::numbers::e * std::numbers::e);
  const int n = 1000;
  const double res = static_cast<double>(std::numbers::e * std::numbers::e) + 1.0;

  // Create function
  std::function<double(double)> _f = [](double x) { return log(x); };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_seq_func_test, equal_borders) {
  const double a = 1;
  const double b = 1;
  const int n = 1000;
  const double res = 0.0;

  // Create function y = x^2
  std::function<double(double)> _f = [](double x) { return x * x; };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_seq_func_test, parabola_with_large_result) {
  const double a = 0;
  const double b = 100;
  const int n = 100000;
  const double res = 1000000.0 / 3.0;

  // Create function y = x^2
  std::function<double(double)> _f = [](double x) { return x * x; };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_seq_func_test, cosinus_result_equals_zero) {
  const double a = 0;
  const auto b = static_cast<double>(std::numbers::pi);
  const int n = 10000;
  const double res = 0.0;

  // Create function y = cos(x)
  std::function<double(double)> _f = [](double x) { return cos(x); };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}

TEST(ivanov_m_integration_trapezoid_seq_func_test, cosinus_result_less_than_zero) {
  const double a = 0;
  const auto b = static_cast<double>(std::numbers::pi);
  const int n = 10000;
  const double res = 0.0;

  // Create function y = cos(x) - 1
  std::function<double(double)> _f = [](double x) { return cos(x); };

  // Create data
  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task

  ivanov_m_integration_trapezoid_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.add_function(_f);

  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_NEAR(res, out[0], 1e-3);
}