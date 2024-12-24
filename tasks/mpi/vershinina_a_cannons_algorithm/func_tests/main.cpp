#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/vershinina_a_cannons_algorithm/include/ops_mpi.hpp"

std::vector<double> getRandomMatrix(double r) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distr(0, 100);
  std::vector<double> matrix(r * r, 0.0);
  for (int i = 0; i < r * r; i++) {
    matrix[i] = distr(gen);
  }
  return matrix;
}

TEST(vershinina_a_cannons_algorithm, Test_1) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  int n = 3;
  auto lhs = getRandomMatrix(3);
  auto rhs = getRandomMatrix(3);

  std::vector<double> res(n * n, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  }
  vershinina_a_cannons_algorithm::TestMPITaskParallel testTaskPar(taskDataPar);
  if (!testTaskPar.validation()) {
    GTEST_SKIP();
  }
  testTaskPar.pre_processing();
  testTaskPar.run();
  testTaskPar.post_processing();
  if (world.rank() == 0) {
    std::vector<double> ref_res(n * n, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_res.data()));

    vershinina_a_cannons_algorithm::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();
    for (int i = 0; i < (int)res.size(); i++) {
      ASSERT_NEAR(res[i], ref_res[i], 0.1);
    }
  }
}

TEST(vershinina_a_cannons_algorithm, Test_2) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  int n = 5;
  auto lhs = getRandomMatrix(5);
  auto rhs = getRandomMatrix(5);

  std::vector<double> res(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  }

  vershinina_a_cannons_algorithm::TestMPITaskParallel testTaskPar(taskDataPar);
  if (!testTaskPar.validation()) {
    GTEST_SKIP();
  }
  testTaskPar.pre_processing();
  testTaskPar.run();
  testTaskPar.post_processing();

  if (world.rank() == 0) {
    std::vector<double> ref_res(n * n, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_res.data()));

    vershinina_a_cannons_algorithm::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)res.size(); i++) {
      ASSERT_NEAR(res[i], ref_res[i], 0.1);
    }
  }
}

TEST(vershinina_a_cannons_algorithm, Test_3) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  int n = 10;
  auto lhs = getRandomMatrix(10);
  auto rhs = getRandomMatrix(10);

  std::vector<double> res(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  }

  vershinina_a_cannons_algorithm::TestMPITaskParallel testTaskPar(taskDataPar);
  if (!testTaskPar.validation()) {
    GTEST_SKIP();
  }
  testTaskPar.pre_processing();
  testTaskPar.run();
  testTaskPar.post_processing();

  if (world.rank() == 0) {
    std::vector<double> ref_res(n * n, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_res.data()));

    vershinina_a_cannons_algorithm::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)res.size(); i++) {
      ASSERT_NEAR(res[i], ref_res[i], 0.1);
    }
  }
}

TEST(vershinina_a_cannons_algorithm, Test_4) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  int n = 15;
  auto lhs = getRandomMatrix(15);
  auto rhs = getRandomMatrix(15);

  std::vector<double> res(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  }

  vershinina_a_cannons_algorithm::TestMPITaskParallel testTaskPar(taskDataPar);
  if (!testTaskPar.validation()) {
    GTEST_SKIP();
  }
  testTaskPar.pre_processing();
  testTaskPar.run();
  testTaskPar.post_processing();

  if (world.rank() == 0) {
    std::vector<double> ref_res(n * n, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_res.data()));

    vershinina_a_cannons_algorithm::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)res.size(); i++) {
      ASSERT_NEAR(res[i], ref_res[i], 0.1);
    }
  }
}
TEST(vershinina_a_cannons_algorithm, Test_5) {
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  int n = 30;
  auto lhs = getRandomMatrix(30);
  auto rhs = getRandomMatrix(30);

  std::vector<double> res(n * n, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
  }

  vershinina_a_cannons_algorithm::TestMPITaskParallel testTaskPar(taskDataPar);
  if (!testTaskPar.validation()) {
    GTEST_SKIP();
  }
  testTaskPar.pre_processing();
  testTaskPar.run();
  testTaskPar.post_processing();

  if (world.rank() == 0) {
    std::vector<double> ref_res(n * n, 0.0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lhs.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(rhs.data()));
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref_res.data()));

    vershinina_a_cannons_algorithm::TestMPITaskSequential testTaskSeq(taskDataSeq);
    ASSERT_TRUE(testTaskSeq.validation());
    testTaskSeq.pre_processing();
    testTaskSeq.run();
    testTaskSeq.post_processing();

    for (int i = 0; i < (int)res.size(); i++) {
      ASSERT_NEAR(res[i], ref_res[i], 0.1);
    }
  }
}
