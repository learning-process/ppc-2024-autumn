#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "mpi/vershinina_a_integration_the_monte_carlo_method/include/ops_mpi.hpp"

std::vector<double> vershinina_a_integration_the_monte_carlo_method::getRandomVector() {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(10, 60);
  std::uniform_int_distribution<> distr_iter(100000, 1000000);
  std::vector<double> vec(5);
  vec[0] = distr(gen);
  vec[1] = vec[0] + distr(gen);
  vec[2] = distr(gen);
  vec[3] = vec[2] + distr(gen);
  vec[4] = distr_iter(gen);
  return vec;
}

TEST(vershinina_a_integration_the_monte_carlo_method, Test_1) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> global_res(1, 0);
  in = vershinina_a_integration_the_monte_carlo_method::getRandomVector();

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  vershinina_a_integration_the_monte_carlo_method::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.p = [](double x) { return exp(sin(4 * x) + 2 * pow(x, 2)); };
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    vershinina_a_integration_the_monte_carlo_method::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.p = [](double x) { return exp(sin(4 * x) + 2 * pow(x, 2)); };
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_NEAR(reference_res[0], global_res[0], 1);
  }
}
TEST(vershinina_a_integration_the_monte_carlo_method, Test_2) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> global_res(1, 0);
  in = vershinina_a_integration_the_monte_carlo_method::getRandomVector();

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  vershinina_a_integration_the_monte_carlo_method::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.p = [](double x) { return exp(sqrt(pow(x, 2) * 2 + x + 1)); };
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    vershinina_a_integration_the_monte_carlo_method::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.p = [](double x) { return exp(sqrt(pow(x, 2) * 2 + x + 1)); };
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_NEAR(reference_res[0], global_res[0], 1);
  }
}
TEST(vershinina_a_integration_the_monte_carlo_method, Test_3) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> global_res(1, 0);
  in = vershinina_a_integration_the_monte_carlo_method::getRandomVector();

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  vershinina_a_integration_the_monte_carlo_method::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.p = [](double x) { return exp(pow(x, 15) * sqrt(1 + 3 * pow(x, 8))); };
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> reference_res(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_res.data()));
    taskDataSeq->outputs_count.emplace_back(reference_res.size());

    vershinina_a_integration_the_monte_carlo_method::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    testMpiTaskSequential.p = [](double x) { return exp(pow(x, 15) * sqrt(1 + 3 * pow(x, 8))); };
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_NEAR(reference_res[0], global_res[0], 1);
  }
}