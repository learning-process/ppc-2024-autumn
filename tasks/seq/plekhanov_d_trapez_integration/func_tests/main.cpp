#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/plekhanov_d_trapez_integration/include/ops_seq.hpp"

TEST(plekhanov_d_trapez_integration_seq, test_int_linear_func) {
  const double a = -1.0;
  const double b = 1.0;
  const int n = 500;
  const double expected = 6.0;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  auto f = [](double x) { return 4 * x + 3; };
  testTaskSequential->set_function(f);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(plekhanov_d_trapez_integration_seq, test_int_sin_func) {
  const double a = 0.0;
  const double b = M_PI;
  const int n = 500;
  const double expected = 2.0;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  auto f = [](double x) { return sin(x); };
  testTaskSequential->set_function(f);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(plekhanov_d_trapez_integration_seq, test_int_exponential_func) {
  const double a = 0.0;
  const double b = 1.0;
  const int n = 500;
  const double expected = exp(1) - 1;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  auto f = [](double x) { return exp(x); };
  testTaskSequential->set_function(f);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(plekhanov_d_trapez_integration_seq, test_int_cubic_func) {
  const double a = 0.0;
  const double b = 1.0;
  const int n = 500;
  const double expected = 0.25;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  auto f = [](double x) { return x * x * x; };
  testTaskSequential->set_function(f);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(plekhanov_d_trapez_integration_seq, test_int_inverse_func) {
  const double a = 0.0;
  const double b = 1.0;
  const int n = 500;
  const double expected = log(2);  // ln(2)

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  auto f = [](double x) { return 1 / (x + 1); };
  testTaskSequential->set_function(f);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(plekhanov_d_trapez_integration_seq, test_int_absolute_func) {
  const double a = -1.0;
  const double b = 1.0;
  const int n = 500;
  const double expected = 1.0;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  auto f = [](double x) { return fabs(x); };
  testTaskSequential->set_function(f);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(plekhanov_d_trapez_integration_seq, test_int_parabolic_func) {
  const double a = 0.0;
  const double b = 2.0;
  const int n = 500;
  const double expected = 16.0 / 3.0;  // 16/3

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  auto f = [](double x) { return -x * x + 4; };
  testTaskSequential->set_function(f);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(plekhanov_d_trapez_integration_seq, test_int_cos_func) {
  const double a = 0.0;
  const double b = M_PI / 2;
  const int n = 500;
  const double expected = 1.0;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<plekhanov_d_trapez_integration_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  auto f = [](double x) { return cos(x); };
  testTaskSequential->set_function(f);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}
