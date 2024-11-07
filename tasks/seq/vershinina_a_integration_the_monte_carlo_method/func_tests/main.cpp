
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "seq/vershinina_a_integration_the_monte_carlo_method/include/ops_seq.hpp"

TEST(vershinina_a_integration_the_monte_carlo_method, test1) {
  std::vector<float> in;
  std::vector<float> reference_res(1, 0);
  const float count_size_vector = 4;
  in = vershinina_a_integration_the_monte_carlo_method::getRandomVector(count_size_vector);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
  taskDataSeq->outputs_count.emplace_back(reference_res.size());

  // Create Task
  vershinina_a_integration_the_monte_carlo_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.p = [](float x) { return exp(sin(4 * x) + 2 * pow(x, 2)); };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count_size_vector, reference_res[0]);
}

TEST(vershinina_a_integration_the_monte_carlo_method, test2) {
  std::vector<float> in;
  std::vector<float> reference_res(1, 0);
  const float count_size_vector = 4;
  in = vershinina_a_integration_the_monte_carlo_method::getRandomVector(count_size_vector);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
  taskDataSeq->outputs_count.emplace_back(reference_res.size());

  // Create Task
  vershinina_a_integration_the_monte_carlo_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.p = [](float x) { return exp(sqrt(pow(x, 2) * 2 + x + 1)); };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count_size_vector, reference_res[0]);
}

TEST(vershinina_a_integration_the_monte_carlo_method, test3) {
  std::vector<float> in;
  std::vector<float> reference_res(1, 0);
  const float count_size_vector = 4;
  in = vershinina_a_integration_the_monte_carlo_method::getRandomVector(count_size_vector);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
  taskDataSeq->outputs_count.emplace_back(reference_res.size());

  vershinina_a_integration_the_monte_carlo_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.p = [](float x) { return exp(pow(x, 15) * sqrt(1 + 3 * pow(x, 8))); };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count_size_vector, reference_res[0]);
}

TEST(vershinina_a_integration_the_monte_carlo_method, test4) {
  std::vector<float> in;
  std::vector<float> reference_res(1, 0);
  const float count_size_vector = 4;
  in = vershinina_a_integration_the_monte_carlo_method::getRandomVector(count_size_vector);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(reference_res.data()));
  taskDataSeq->outputs_count.emplace_back(reference_res.size());

  vershinina_a_integration_the_monte_carlo_method::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.p = [](float x) { return exp(pow(x, 8) / sqrt(7 * pow(x, 4) + 25)); };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(count_size_vector, reference_res[0]);
}