#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdint>
#include <cstdio>
#include <numbers>
#include <vector>

#include "mpi/vedernikova_k_monte_carlo_integration_mpi/include/ops_mpi.hpp"

TEST(vedernikova_k_monte_carlo_integration_mpi, number_of_points_500000_seq) {
  auto [ax, bx] = std::make_pair(0.0, 2.0);
  auto [ay, by] = std::make_pair(0.0, std::numbers::pi);
  auto [az, bz] = std::make_pair(0.0, std::numbers::pi);

  size_t num_point = 500000;
  double out = 0.0;
  double expected_res = 128 * std::numbers::pi / 15;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&az));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bz));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

  taskDataSeq->inputs_count.emplace_back(taskDataSeq->inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vedernikova_k_monte_carlo_integration_mpi::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.f = [](double x, double y, double z) { return x * x + y * y; };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(expected_res, out, 1.0);
}

TEST(vedernikova_k_monte_carlo_integration_mpi, number_of_points_600000_seq) {
  auto [ax, bx] = std::make_pair(0.0, 2.0);
  auto [ay, by] = std::make_pair(0.0, std::numbers::pi);
  auto [az, bz] = std::make_pair(0.0, std::numbers::pi);
  size_t num_point = 600000;

  double out = 0.0;
  double expected_res = 128 * std::numbers::pi / 15;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&az));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bz));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

  taskDataSeq->inputs_count.emplace_back(taskDataSeq->inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vedernikova_k_monte_carlo_integration_mpi::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.f = [](double x, double y, double z) { return x * x + y * y; };
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  EXPECT_NEAR(expected_res, out, 1.0);
}
TEST(vedernikova_k_monte_carlo_integration_mpi, validation_false) {
  auto [ax, bx] = std::make_pair(0.0, 2.0);
  auto [ay, by] = std::make_pair(0.0, std::numbers::pi);
  auto [az, bz] = std::make_pair(0.0, std::numbers::pi);
  // size_t num_point = 1000000;

  double out = 0.0;

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&az));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bz));
  // taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));    <--missed

  taskDataSeq->inputs_count.emplace_back(taskDataSeq->inputs.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  taskDataSeq->outputs_count.emplace_back(1);

  // Create Task
  vedernikova_k_monte_carlo_integration_mpi::TestTaskSequential testTaskSequential(taskDataSeq);
  testTaskSequential.f = [](double x, double y, double z) { return x * x + y * y; };
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(vedernikova_k_monte_carlo_integration_mpi, number_of_points_1000000_mpi) {
  boost::mpi::communicator world;
  auto [ax, bx] = std::make_pair(0.0, 2.0);
  auto [ay, by] = std::make_pair(0.0, std::numbers::pi);
  auto [az, bz] = std::make_pair(0.0, std::numbers::pi);
  size_t num_point = 1000000;

  double out = 0.0;
  double expected_res = 128 * std::numbers::pi / 15;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&az));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bz));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  }

  vedernikova_k_monte_carlo_integration_mpi::TestMPITaskParallel testTaskMPI(taskDataPar);
  testTaskMPI.f = [](double x, double y, double z) { return x * x + y * y; };
  ASSERT_EQ(testTaskMPI.validation(), true);
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();
  if (world.rank() == 0) {
    EXPECT_NEAR(expected_res, out, 1.0);
  }
}
TEST(vedernikova_k_monte_carlo_integration_mpi, number_of_points_1500000_mpi) {
  boost::mpi::communicator world;
  auto [ax, bx] = std::make_pair(0.0, 2.0);
  auto [ay, by] = std::make_pair(0.0, std::numbers::pi);
  auto [az, bz] = std::make_pair(0.0, std::numbers::pi);
  size_t num_point = 1500000;

  double out = 0.0;
  double expected_res = 128 * std::numbers::pi / 15;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&az));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bz));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  }

  vedernikova_k_monte_carlo_integration_mpi::TestMPITaskParallel testTaskMPI(taskDataPar);
  testTaskMPI.f = [](double x, double y, double z) { return x * x + y * y; };
  ASSERT_EQ(testTaskMPI.validation(), true);
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();
  if (world.rank() == 0) {
    EXPECT_NEAR(expected_res, out, 1.0);
  }
}
TEST(vedernikova_k_monte_carlo_integration_mpi, number_of_points_2000000_mpi) {
  boost::mpi::communicator world;
  auto [ax, bx] = std::make_pair(0.0, 2.0);
  auto [ay, by] = std::make_pair(0.0, std::numbers::pi);
  auto [az, bz] = std::make_pair(0.0, std::numbers::pi);
  size_t num_point = 2000000;

  double out = 0.0;
  double expected_res = 128 * std::numbers::pi / 15;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ax));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bx));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&ay));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&by));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&az));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bz));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&num_point));

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));
  }

  vedernikova_k_monte_carlo_integration_mpi::TestMPITaskParallel testTaskMPI(taskDataPar);
  testTaskMPI.f = [](double x, double y, double z) { return x * x + y * y; };
  ASSERT_EQ(testTaskMPI.validation(), true);
  testTaskMPI.pre_processing();
  testTaskMPI.run();
  testTaskMPI.post_processing();
  if (world.rank() == 0) {
    EXPECT_NEAR(expected_res, out, 1.0);
  }
}
