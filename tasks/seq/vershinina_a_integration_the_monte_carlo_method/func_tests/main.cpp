#include <gtest/gtest.h>

#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "seq/vershinina_a_integration_the_monte_carlo_method/include/ops_seq.hpp"

std::vector<double> vershinina_a_integration_the_monte_carlo_method::getRandomVector() {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(10, 60);
  std::uniform_int_distribution<> distr_iter(10000, 100000);
  std::vector<double> vec(5);
  vec[0] = distr(gen);
  vec[1] = vec[0] + distr(gen);
  vec[2] = distr(gen);
  vec[3] = vec[2] + distr(gen);
  vec[4] = distr_iter(gen);
  return vec;
}

TEST(vershinina_a_integration_the_monte_carlo_method, test1) {
  std::vector<double> in{5, 15, 0, 100, 100000};
  std::vector<double> reference_res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
  taskDataSeq->outputs_count.emplace_back(reference_res.size());

  // Create Task
  vershinina_a_integration_the_monte_carlo_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.p = [](double x) { return exp(sin(4 * x) + 2 * pow(x, 2)); };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(reference_res[0], 1000, 1);
}

TEST(vershinina_a_integration_the_monte_carlo_method, test2) {
  std::vector<double> in{6, 13, 0, 14, 100000};
  std::vector<double> reference_res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
  taskDataSeq->outputs_count.emplace_back(reference_res.size());

  // Create Task
  vershinina_a_integration_the_monte_carlo_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.p = [](double x) { return exp(sqrt(pow(x, 2) * 2 + x + 1)); };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(reference_res[0], 98, 1);
}

TEST(vershinina_a_integration_the_monte_carlo_method, test3) {
  std::vector<double> in{-2, 4, -2, 4, 100000};
  std::vector<double> reference_res(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
  taskDataSeq->outputs_count.emplace_back(reference_res.size());

  vershinina_a_integration_the_monte_carlo_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.p = [](double x) { return exp(8 + 2 * x - pow(x, 2)); };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(reference_res[0], 35, 1);
}

TEST(vershinina_a_integration_the_monte_carlo_method, test4) {
  std::vector<double> in;
  std::vector<double> reference_res(1, 0);
  double xmin{};
  double xmax{};

  std::vector<double> another_res(1, 0);

  in = vershinina_a_integration_the_monte_carlo_method::getRandomVector();
  xmin = in[0];
  xmax = in[1];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
  taskDataSeq->outputs_count.emplace_back(reference_res.size());

  vershinina_a_integration_the_monte_carlo_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.p = [](double x) { return exp(sin(x)); };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  another_res[0] = (sin(xmax) - sin(xmin));
  EXPECT_NEAR(another_res[0], reference_res[0], 10);
}

TEST(vershinina_a_integration_the_monte_carlo_method, test5) {
  std::vector<double> in;
  std::vector<double> reference_res(1, 0);
  double xmin{};
  double xmax{};

  std::vector<double> another_res(1, 0);

  in = vershinina_a_integration_the_monte_carlo_method::getRandomVector();
  xmin = in[0];
  xmax = in[1];

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
  taskDataSeq->outputs_count.emplace_back(reference_res.size());

  vershinina_a_integration_the_monte_carlo_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.p = [](double x) { return exp(cos(x)); };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  another_res[0] = (cos(xmax) - cos(xmin));
  EXPECT_NEAR(another_res[0], reference_res[0], 10);
}