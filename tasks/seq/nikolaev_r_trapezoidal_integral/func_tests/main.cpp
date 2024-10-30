#define _USE_MATH_DEFINES

#include <gtest/gtest.h>
#include <vector>
#include <cmath>

#include "seq/nikolaev_r_trapezoidal_integral/include/ops_seq.hpp"

TEST(nikolaev_r_trapezoidal_integral_seq, test_int_linear_func) {
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
	  std::make_shared<nikolaev_r_trapezoidal_integral_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return 4 * x + 3; });
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(nikolaev_r_trapezoidal_integral_seq, test_int_squared_func) {
  const double a = 0.0;
  const double b = 2.0;
  const int n = 500;
  const double expected = 29.33;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_trapezoidal_integral_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return 5 * x * x + 8; });
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(nikolaev_r_trapezoidal_integral_seq, test_int_4th_degree_pol_func) {
  const double a = -2.0;
  const double b = 2.0;
  const int n = 500;
  const double expected = -5.87;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_trapezoidal_integral_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return pow(x, 4) + 3 * pow(x, 3) - 5 * x * x + 2; });
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(nikolaev_r_trapezoidal_integral_seq, test_int_exp_func) {
  const double a = 0.0;
  const double b = 1.0;
  const int n = 500;
  const double expected = 6.36;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_trapezoidal_integral_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return exp(3 * x); });
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(nikolaev_r_trapezoidal_integral_seq, test_int_pow_func) {
  const double a = 0.0;
  const double b = 3.0;
  const int n = 500;
  const double expected = 63.44;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_trapezoidal_integral_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return pow(4, x) + 6; });
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(nikolaev_r_trapezoidal_integral_seq, test_int_cosine_func) {
  const double a = M_PI / 2;
  const double b = M_PI * 2;
  const int n = 500;
  const double expected = -1.0;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_trapezoidal_integral_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return cos(x); });
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(nikolaev_r_trapezoidal_integral_seq, test_int_sin_func) {
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
      std::make_shared<nikolaev_r_trapezoidal_integral_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return sin(x); });
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}

TEST(nikolaev_r_trapezoidal_integral_seq, test_int_mixed_func) {
  const double a = 1;
  const double b = 3;
  const int n = 500;
  const double expected = 83.19;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential =
      std::make_shared<nikolaev_r_trapezoidal_integral_seq::TrapezoidalIntegralSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return 3 * x * x + pow(5, x) - exp(x); });
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  ASSERT_NEAR(expected, out[0], 0.01);
}