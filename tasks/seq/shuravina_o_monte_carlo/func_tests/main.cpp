#include <gtest/gtest.h>

#include <vector>

#include "seq/shuravina_o_monte_carlo/include/ops_seq.hpp"

TEST(MonteCarloIntegrationTaskSequential, Test_Integration) {
  std::vector<double> out(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(nullptr);
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskSequential>(taskDataSeq);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  double expected_integral = 1.0 / 3.0;
  ASSERT_NEAR(expected_integral, out[0], 0.01);
}

TEST(MonteCarloIntegrationTaskSequential, Test_Boundary_Conditions) {
  std::vector<double> out(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(nullptr);
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskSequential>(taskDataSeq);
  testTaskSequential->set_interval(-1.0, 1.0);
  testTaskSequential->set_num_points(1000000);
  testTaskSequential->set_function([](double x) { return x * x; });

  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  double expected_integral = 2.0 / 3.0;
  ASSERT_NEAR(expected_integral, out[0], 0.01);
}