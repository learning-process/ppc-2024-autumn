#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <climits>
#include <vector>

#include "mpi/malyshev_a_two_dim_global_optimization_characteristics/include/ops_mpi.hpp"

TEST(malyshev_a_two_dim_global_optimization_characteristics_mpi, SimpleTest) {
  boost::mpi::communicator world;
  auto target = [](double x, double y) -> double { return pow(x - 2, 2) + pow(y - 3, 2); };

  std::vector<malyshev_a_two_dim_global_optimization_characteristics_mpi::constraint_t> constraints{
      [](double x, double y) -> bool { return x + y >= 4; }, [](double x, double y) -> bool { return x - y <= 1; },
      [](double x, double y) -> bool { return x >= 0; }, [](double x, double y) -> bool { return y >= 0; }};

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel taskMPI(taskDataMPI, target,
                                                                                       constraints);
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential taskSeq(taskDataSeq, target,
                                                                                         constraints);

  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resSeq;
  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resMPI;
  std::pair<double, double> bounds[2]{{0, 10}, {0, 10}};
  double eps = 1e-4;

  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resMPI));
    taskDataMPI->inputs_count.emplace_back(2);
    taskDataMPI->inputs_count.emplace_back(1);
    taskDataMPI->outputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resSeq));
    taskDataSeq->inputs = taskDataMPI->inputs;
    taskDataSeq->inputs_count = taskDataMPI->inputs_count;
    taskDataSeq->outputs_count = taskDataMPI->outputs_count;
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    ASSERT_NEAR(resMPI.x, resSeq.x, eps);
    ASSERT_NEAR(resMPI.y, resSeq.y, eps);
    ASSERT_NEAR(resMPI.value, resSeq.value, eps);
  }
}

TEST(malyshev_a_two_dim_global_optimization_characteristics_mpi, RastrigrinTest) {
  boost::mpi::communicator world;
  auto target = [](double x, double y) -> double {
    return 20 + (pow(x, 2) - 10 * cos(2 * M_PI * x)) + (pow(y, 2) - 10 * cos(2 * M_PI * y));
  };

  std::vector<malyshev_a_two_dim_global_optimization_characteristics_mpi::constraint_t> constraints{
      [](double x, double y) { return x + y >= -1; }, [](double x, double y) { return x - y <= 2; },
      [](double x, double y) { return x >= 0; }, [](double x, double y) { return y >= 0; }};

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel taskMPI(taskDataMPI, target,
                                                                                       constraints);
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential taskSeq(taskDataSeq, target,
                                                                                         constraints);

  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resSeq;
  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resMPI;
  std::pair<double, double> bounds[2]{{-4.12, 4.12}, {-4.12, 4.12}};
  double eps = 1e-4;

  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resMPI));
    taskDataMPI->inputs_count.emplace_back(2);
    taskDataMPI->inputs_count.emplace_back(1);
    taskDataMPI->outputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resSeq));
    taskDataSeq->inputs = taskDataMPI->inputs;
    taskDataSeq->inputs_count = taskDataMPI->inputs_count;
    taskDataSeq->outputs_count = taskDataMPI->outputs_count;
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    ASSERT_NEAR(resMPI.x, resSeq.x, eps);
    ASSERT_NEAR(resMPI.y, resSeq.y, eps);
    ASSERT_NEAR(resMPI.value, resSeq.value, eps);
  }
}

TEST(malyshev_a_two_dim_global_optimization_characteristics_mpi, HimmelblauTest) {
  boost::mpi::communicator world;
  auto target = [](double x, double y) -> double { return pow(x * x + y - 11, 2) + pow(x + y * y - 7, 2); };

  std::vector<malyshev_a_two_dim_global_optimization_characteristics_mpi::constraint_t> constraints{
      [](double x, double y) { return x * x + y * y <= 16; }, [](double x, double y) { return x >= -4; },
      [](double x, double y) { return y >= -4; }};

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel taskMPI(taskDataMPI, target,
                                                                                       constraints);
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential taskSeq(taskDataSeq, target,
                                                                                         constraints);

  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resSeq;
  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resMPI;
  std::pair<double, double> bounds[2]{{-4.0, 4.0}, {-4.0, 4.0}};
  double eps = 1e-4;

  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resMPI));
    taskDataMPI->inputs_count.emplace_back(2);
    taskDataMPI->inputs_count.emplace_back(1);
    taskDataMPI->outputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resSeq));
    taskDataSeq->inputs = taskDataMPI->inputs;
    taskDataSeq->inputs_count = taskDataMPI->inputs_count;
    taskDataSeq->outputs_count = taskDataMPI->outputs_count;
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    ASSERT_NEAR(resMPI.x, resSeq.x, eps);
    ASSERT_NEAR(resMPI.y, resSeq.y, eps);
    ASSERT_NEAR(resMPI.value, resSeq.value, eps);
  }
}

TEST(malyshev_a_two_dim_global_optimization_characteristics_mpi, AckleyTest) {
  boost::mpi::communicator world;
  auto target = [](double x, double y) -> double {
    return -20 * exp(-0.2 * sqrt(0.5 * (x * x + y * y))) - exp(0.5 * (cos(2 * M_PI * x) + cos(2 * M_PI * y))) + M_E +
           20;
  };

  std::vector<malyshev_a_two_dim_global_optimization_characteristics_mpi::constraint_t> constraints{
      [](double x, double y) { return x * x + y * y <= 25; }, [](double x, double y) { return x >= -5; },
      [](double x, double y) { return y >= -5; }};

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel taskMPI(taskDataMPI, target,
                                                                                       constraints);
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential taskSeq(taskDataSeq, target,
                                                                                         constraints);

  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resSeq;
  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resMPI;
  std::pair<double, double> bounds[2]{{-5.0, 5.0}, {-5.0, 5.0}};
  double eps = 1e-4;

  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resMPI));
    taskDataMPI->inputs_count.emplace_back(2);
    taskDataMPI->inputs_count.emplace_back(1);
    taskDataMPI->outputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resSeq));
    taskDataSeq->inputs = taskDataMPI->inputs;
    taskDataSeq->inputs_count = taskDataMPI->inputs_count;
    taskDataSeq->outputs_count = taskDataMPI->outputs_count;
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    ASSERT_NEAR(resMPI.x, resSeq.x, eps);
    ASSERT_NEAR(resMPI.y, resSeq.y, eps);
    ASSERT_NEAR(resMPI.value, resSeq.value, eps);
  }
}

TEST(malyshev_a_two_dim_global_optimization_characteristics_mpi, MatyasTest) {
  boost::mpi::communicator world;
  auto target = [](double x, double y) -> double { return 0.26 * (x * x + y * y) - 0.48 * x * y; };

  std::vector<malyshev_a_two_dim_global_optimization_characteristics_mpi::constraint_t> constraints{
      [](double x, double y) { return x * x + y * y <= 16; }, [](double x, double y) { return x >= -4; },
      [](double x, double y) { return y >= -4; }};

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel taskMPI(taskDataMPI, target,
                                                                                       constraints);
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential taskSeq(taskDataSeq, target,
                                                                                         constraints);

  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resSeq;
  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resMPI;
  std::pair<double, double> bounds[2]{{-4.0, 4.0}, {-4.0, 4.0}};
  double eps = 1e-4;

  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resMPI));
    taskDataMPI->inputs_count.emplace_back(2);
    taskDataMPI->inputs_count.emplace_back(1);
    taskDataMPI->outputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resSeq));
    taskDataSeq->inputs = taskDataMPI->inputs;
    taskDataSeq->inputs_count = taskDataMPI->inputs_count;
    taskDataSeq->outputs_count = taskDataMPI->outputs_count;
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    ASSERT_NEAR(resMPI.x, resSeq.x, eps);
    ASSERT_NEAR(resMPI.y, resSeq.y, eps);
    ASSERT_NEAR(resMPI.value, resSeq.value, eps);
  }
}

TEST(malyshev_a_two_dim_global_optimization_characteristics_mpi, GoldsteinPriceTest) {
  boost::mpi::communicator world;
  auto target = [](double x, double y) -> double {
    return (1 + pow(x + y + 1, 2) * (19 - 14 * x + 3 * x * x - 14 * y + 6 * x * y + 3 * y * y)) *
           (30 + pow(2 * x - 3 * y, 2) * (18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y));
  };

  std::vector<malyshev_a_two_dim_global_optimization_characteristics_mpi::constraint_t> constraints{
      [](double x, double y) { return x * x + y * y <= 4; }, [](double x, double y) { return x >= -2; },
      [](double x, double y) { return y >= -2; }};

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel taskMPI(taskDataMPI, target,
                                                                                       constraints);
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential taskSeq(taskDataSeq, target,
                                                                                         constraints);

  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resSeq;
  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resMPI;
  std::pair<double, double> bounds[2]{{-2.0, 2.0}, {-2.0, 2.0}};
  double eps = 1e-4;

  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resMPI));
    taskDataMPI->inputs_count.emplace_back(2);
    taskDataMPI->inputs_count.emplace_back(1);
    taskDataMPI->outputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resSeq));
    taskDataSeq->inputs = taskDataMPI->inputs;
    taskDataSeq->inputs_count = taskDataMPI->inputs_count;
    taskDataSeq->outputs_count = taskDataMPI->outputs_count;
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    ASSERT_NEAR(resMPI.x, resSeq.x, eps);
    ASSERT_NEAR(resMPI.y, resSeq.y, eps);
    ASSERT_NEAR(resMPI.value, resSeq.value, eps);
  }
}

TEST(malyshev_a_two_dim_global_optimization_characteristics_mpi, BoothTest) {
  boost::mpi::communicator world;
  auto target = [](double x, double y) -> double { return pow(x + 2 * y - 7, 2) + pow(2 * x + y - 5, 2); };

  std::vector<malyshev_a_two_dim_global_optimization_characteristics_mpi::constraint_t> constraints{
      [](double x, double y) { return x >= -10; }, [](double x, double y) { return y >= -10; },
      [](double x, double y) { return x * x + y * y <= 100; }};

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskParallel taskMPI(taskDataMPI, target,
                                                                                       constraints);
  malyshev_a_two_dim_global_optimization_characteristics_mpi::TestTaskSequential taskSeq(taskDataSeq, target,
                                                                                         constraints);

  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resSeq;
  malyshev_a_two_dim_global_optimization_characteristics_mpi::Point resMPI;
  std::pair<double, double> bounds[2]{{-10.0, 10.0}, {-10.0, 10.0}};
  double eps = 1e-4;

  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&bounds));
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(&eps));
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resMPI));
    taskDataMPI->inputs_count.emplace_back(2);
    taskDataMPI->inputs_count.emplace_back(1);
    taskDataMPI->outputs_count.emplace_back(1);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&resSeq));
    taskDataSeq->inputs = taskDataMPI->inputs;
    taskDataSeq->inputs_count = taskDataMPI->inputs_count;
    taskDataSeq->outputs_count = taskDataMPI->outputs_count;
  }

  ASSERT_TRUE(taskMPI.validation());
  ASSERT_TRUE(taskMPI.pre_processing());
  ASSERT_TRUE(taskMPI.run());
  ASSERT_TRUE(taskMPI.post_processing());

  if (world.rank() == 0) {
    ASSERT_TRUE(taskSeq.validation());
    ASSERT_TRUE(taskSeq.pre_processing());
    ASSERT_TRUE(taskSeq.run());
    ASSERT_TRUE(taskSeq.post_processing());

    ASSERT_NEAR(resMPI.x, resSeq.x, eps);
    ASSERT_NEAR(resMPI.y, resSeq.y, eps);
    ASSERT_NEAR(resMPI.value, resSeq.value, eps);
  }
}