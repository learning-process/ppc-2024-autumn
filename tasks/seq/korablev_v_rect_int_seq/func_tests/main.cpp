#include <gtest/gtest.h>
#define _USE_MATH_DEFINES
#include <cmath>
#include <memory>
#include <vector>

#include "seq/korablev_v_rect_int/include/ops_seq.hpp"

TEST(korablev_v_rectangular_integration_seq, test_integration_x_squared) {
  const double a = 0.0;
  const double b = 1.0;
  const int n = 1000;
  const double expected_result = 1.0 / 3.0;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<korablev_v_rect_int_seq::RectangularIntegraitionSequential>(taskDataSeq);

  std::function<double(double)> func = [](double x) { return x * x; };
  testTaskSequential->set_function(func);

  ASSERT_TRUE(testTaskSequential->validation());
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(korablev_v_rectangular_integration_seq, test_integration_x) {
  const double a = 0.0;
  const double b = 1.0;
  const int n = 1000;

  const double expected_result = 0.5;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_rect_int_seq::RectangularIntegraitionSequential testTaskSequential(taskDataSeq);

  std::function<double(double)> func = [](double x) { return x; };
  testTaskSequential.set_function(func);

  ASSERT_TRUE(testTaskSequential.validation());

  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(korablev_v_rectangular_integration_seq, test_integration_sin_x) {
  const double a = 0.0;
  const double b = M_PI;
  const int n = 1000;

  const double expected_result = 2.0;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_rect_int_seq::RectangularIntegraitionSequential testTaskSequential(taskDataSeq);

  std::function<double(double)> func = [](double x) { return std::sin(x); };
  testTaskSequential.set_function(func);

  ASSERT_TRUE(testTaskSequential.validation());

  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}

TEST(korablev_v_rectangular_integration_seq, test_integration_exp_x) {
  const double a = 0.0;
  const double b = 1.0;
  const int n = 1000;

  const double expected_result = std::exp(1.0) - 1.0;

  std::vector<double> in = {a, b, static_cast<double>(n)};
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  korablev_v_rect_int_seq::RectangularIntegraitionSequential testTaskSequential(taskDataSeq);

  std::function<double(double)> func = [](double x) { return std::exp(x); };
  testTaskSequential.set_function(func);

  ASSERT_TRUE(testTaskSequential.validation());

  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();

  ASSERT_NEAR(out[0], expected_result, 1e-3);
}