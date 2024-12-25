#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "seq/prokhorov_n_rectangular_integration/include/ops_seq.hpp"

TEST(prokhorov_n_rectangular_integration_seq, test_integration_cos_x) {
  const double lower_bound = 0.0;
  const double upper_bound = M_PI / 2.0;
  const int n = 1000;
  const double expected_result = 1.0;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return std::cos(x); });

  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_x_cubed) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 1000;
  const double expected_result = 0.25;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return x * x * x; });

  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_sqrt_x) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 1000;
  const double expected_result = 2.0 / 3.0;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return std::sqrt(x); });

  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_one_over_x) {
  const double lower_bound = 1.0;
  const double upper_bound = 2.0;
  const int n = 1000;
  const double expected_result = std::log(2.0);

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return 1.0 / x; });

  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_sin_squared_x) {
  const double lower_bound = 0.0;
  const double upper_bound = M_PI;
  const int n = 1000;
  const double expected_result = M_PI / 2.0;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return std::sin(x) * std::sin(x); });

  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}
TEST(prokhorov_n_rectangular_integration_seq, test_integration_exp_minus_x_squared) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 1000;
  const double expected_result = 0.746824;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return std::exp(-x * x); });

  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}
TEST(prokhorov_n_rectangular_integration_seq, test_integration_log_x) {
  const double lower_bound = 1.0;
  const double upper_bound = 2.0;
  const int n = 1000;
  const double expected_result = 2.0 * std::log(2.0) - 1.0;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return std::log(x); });

  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_x_sin_x) {
  const double lower_bound = 0.0;
  const double upper_bound = M_PI;
  const int n = 1000;
  const double expected_result = M_PI;

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return x * std::sin(x); });

  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(prokhorov_n_rectangular_integration_seq, test_integration_atan_x) {
  const double lower_bound = 0.0;
  const double upper_bound = 1.0;
  const int n = 1000;
  const double expected_result = M_PI / 4.0 - 0.5 * std::log(2.0);

  std::vector<double> in = {lower_bound, upper_bound, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<prokhorov_n_rectangular_integration::TestTaskSequential>(taskDataSeq);
  testTaskSequential->set_function([](double x) { return std::atan(x); });

  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}