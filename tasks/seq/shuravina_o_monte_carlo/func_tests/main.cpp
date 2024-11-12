#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
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
TEST(MonteCarloIntegrationTaskSequential, Test_Execution_Time) {
  std::vector<double> out(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(nullptr);
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskSequential>(taskDataSeq);
  ASSERT_EQ(testTaskSequential->validation(), true);

  auto start_time = std::chrono::high_resolution_clock::now();
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> elapsed_time = end_time - start_time;
  std::cout << "Execution time: " << elapsed_time.count() << " seconds" << std::endl;
  ASSERT_LT(elapsed_time.count(), 10.0);
}

TEST(MonteCarloIntegrationTaskSequential, Test_Large_Number_Of_Points) {
  std::vector<double> out(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(nullptr);
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto testTaskSequential = std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskSequential>(taskDataSeq);
  testTaskSequential->set_num_points(10000000);
  ASSERT_EQ(testTaskSequential->validation(), true);
  testTaskSequential->pre_processing();
  testTaskSequential->run();
  testTaskSequential->post_processing();

  double expected_integral = 1.0 / 3.0;
  ASSERT_NEAR(expected_integral, out[0], 0.001);
}

TEST(MonteCarloIntegrationTaskSequential, Test_Validation_Failure) {
  std::vector<double> out(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(nullptr);
  taskDataSeq->outputs_count.emplace_back(1);

  auto testTaskSequential = std::make_shared<shuravina_o_monte_carlo::MonteCarloIntegrationTaskSequential>(taskDataSeq);
  ASSERT_EQ(testTaskSequential->validation(), false);
}