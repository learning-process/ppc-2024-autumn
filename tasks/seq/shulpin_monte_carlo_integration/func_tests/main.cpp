#include <gtest/gtest.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "seq/shulpin_monte_carlo_integration/include/monte_carlo_integral.hpp"

constexpr double ESTIMATE = 1e-3;

TEST(shulpin_monte_carlo_integration, sin_test) {
  double lower_limit = 0.0;
  double upper_limit = M_PI;
  int interval_count = 100000;
  const double expected_sin_integral_result = 2.0;

  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> seq_sin_task_data = std::make_shared<ppc::core::TaskData>();

  seq_sin_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
  seq_sin_task_data->inputs_count.emplace_back(1);

  seq_sin_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
  seq_sin_task_data->inputs_count.emplace_back(1);

  seq_sin_task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(&interval_count));
  seq_sin_task_data->inputs_count.emplace_back(1);

  seq_sin_task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  seq_sin_task_data->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<shulpin_monte_carlo_integration::TestMPITaskSequential>(seq_sin_task_data);

  testTaskSequential->set_seq(shulpin_monte_carlo_integration::fsin);

  ASSERT_TRUE(testTaskSequential->validation());
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(output, expected_sin_integral_result, ESTIMATE);
}

TEST(shulpin_monte_carlo_integration, test_cos) {
  const double lower_limit = 0.0;
  const double upper_limit = M_PI;
  const int interval_count = 100000;
  const double expected_cos_integral_result = 0.0;

  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(&lower_limit)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(&upper_limit)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&interval_count)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  taskDataSeq->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<shulpin_monte_carlo_integration::TestMPITaskSequential>(taskDataSeq);

  testTaskSequential->set_seq(shulpin_monte_carlo_integration::fcos);

  ASSERT_TRUE(testTaskSequential->validation());
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(output, expected_cos_integral_result, ESTIMATE);
}

TEST(shulpin_monte_carlo_integration, test_two_sin_cos) {
  const double lower_limit = 0.0;
  const double upper_limit = M_PI;
  const int interval_count = 100000;
  const double expected_integral_result = 0.0;
  double output = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(&lower_limit)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(&upper_limit)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<int*>(&interval_count)));
  taskDataSeq->inputs_count.emplace_back(1);

  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
  taskDataSeq->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<shulpin_monte_carlo_integration::TestMPITaskSequential>(taskDataSeq);

  testTaskSequential->set_seq(shulpin_monte_carlo_integration::f_two_sin_cos);

  ASSERT_TRUE(testTaskSequential->validation());
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  ASSERT_NEAR(output, expected_integral_result, ESTIMATE);
}
