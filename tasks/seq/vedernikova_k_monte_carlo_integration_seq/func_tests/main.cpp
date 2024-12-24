#include <gtest/gtest.h>

#include <cstddef>
#include <functional>
#include <vector>

#include "seq/vedernikova_k_monte_carlo_integration_seq/include/ops_seq.hpp"

TEST(vedernikova_k_monte_carlo_integration_seq, number_of_points_300000) {
  double ax = -1.0;
  double bx = 1.0;
  double ay = -2.0;
  double by = 2.0;
  size_t num_point = 300000;

  double out = 0.0;
  double expected_res = 8.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

  taskDataSeq->inputs_count.emplace_back(taskDataSeq->inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vedernikova_k_monte_carlo_integration_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.f = [](double x, double y) { return 1 - (1 / 3) * x - 0.25 * y; };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(expected_res, out, 1.0);
}

TEST(vedernikova_k_monte_carlo_integration_seq, number_of_points_400000) {
  double ax = -1.0;
  double bx = 1.0;
  double ay = -2.0;
  double by = 2.0;
  size_t num_point = 400000;

  double out = 0.0;
  double expected_res = 8.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

  taskDataSeq->inputs_count.emplace_back(taskDataSeq->inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vedernikova_k_monte_carlo_integration_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.f = [](double x, double y) { return 1 - (1 / 3) * x - 0.25 * y; };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(expected_res, out, 1.0);
}

TEST(vedernikova_k_monte_carlo_integration_seq, number_of_points_500000) {
  double ax = -1.0;
  double bx = 1.0;
  double ay = -2.0;
  double by = 2.0;
  size_t num_point = 500000;

  double out = 0.0;
  double expected_res = 8.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

  taskDataSeq->inputs_count.emplace_back(taskDataSeq->inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vedernikova_k_monte_carlo_integration_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.f = [](double x, double y) { return 1 - (1 / 3) * x - 0.25 * y; };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(expected_res, out, 1.0);
}
TEST(vedernikova_k_monte_carlo_integration_seq, validation_false) {
  double ax = -1.0;
  double bx = 1.0;
  double ay = -2.0;
  double by = 2.0;

  double out = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
  // taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));   <-- missed

  taskDataSeq->inputs_count.emplace_back(taskDataSeq->inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vedernikova_k_monte_carlo_integration_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.f = [](double x, double y) { return 1 - (1 / 3) * x - 0.25 * y; };
  ASSERT_EQ(testTaskSequential.validation(), false);
}
