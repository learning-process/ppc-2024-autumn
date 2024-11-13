// Golovkin Maksim
#define _USE_MATH_DEFINES
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

#include "mpi/golovkin_integration_rectangular_method/include/ops_mpi.hpp"

using namespace std::chrono_literals;
using namespace golovkin_integration_rectangular_method;

TEST(golovkin_integration_rectangular_method, test_constant_function) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = 10.0;
  int partition_count = 1000000;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }
  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return 5.0; });
  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();
  if (world.rank() >= 4) {
    std::vector<double> expected_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return 5.0; });
    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();
    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_linear_function) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = 5.0;
  int partition_count = 1000000;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return 2.0 * x + 3.0; });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    std::vector<double> expected_result(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return 2.0 * x + 3.0; });

    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();

    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_quadratic_function) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = -3.0;
  double upper_limit = 3.0;
  int partition_count = 1000000;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return x * x; });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();
  if (world.size() < 5 || world.rank() >= 4) {
    std::vector<double> expected_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return x * x; });

    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();

    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_sine_function) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = M_PI;
  int partition_count = 1000000;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return std::sin(x); });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    std::vector<double> expected_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return std::sin(x); });

    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();

    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_cosine_function) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = M_PI / 2;
  int partition_count = 1000000;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return std::cos(x); });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    std::vector<double> expected_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return std::cos(x); });

    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();

    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_exponential_function) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = 1.0;
  int partition_count = 1000000;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return std::exp(x); });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    std::vector<double> expected_result(1, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(expected_result.data()));
    taskDataSeq->outputs_count.emplace_back(expected_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator sequential_task(taskDataSeq);
    sequential_task.set_function([](double x) { return std::exp(x); });

    ASSERT_EQ(sequential_task.validation(), true);
    sequential_task.pre_processing();
    sequential_task.run();
    sequential_task.post_processing();

    ASSERT_NEAR(expected_result[0], computed_result[0], 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_negative_range) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 5.0;
  double upper_limit = -5.0;
  int partition_count = 1000000;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return x; });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    ASSERT_LT(computed_result[0], 0);
  }
}

TEST(golovkin_integration_rectangular_method, test_zero_partitions) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = 10.0;
  int partition_count = 0;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return 5.0; });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    ASSERT_EQ(computed_result[0], 0);
  }
}

TEST(golovkin_integration_rectangular_method, test_large_range) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = -1e6;
  double upper_limit = 1e6;
  int partition_count = 1000000;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return x; });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    ASSERT_NEAR(computed_result[0], 0, 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_small_interval_near_zero) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 1e-10;
  double upper_limit = 1e-5;
  int partition_count = 1000000;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator parallel_task(taskDataPar);
  parallel_task.set_function([](double x) { return x * x; });

  ASSERT_EQ(parallel_task.validation(), true);
  parallel_task.pre_processing();
  parallel_task.run();
  parallel_task.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    double expected_result = (std::pow(upper_limit, 3) - std::pow(lower_limit, 3)) / 3;
    ASSERT_NEAR(computed_result[0], expected_result, 1e-10);
  }
}
TEST(golovkin_integration_rectangular_method, test_validation) {
  boost::mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  int dummy_inputs_count[2] = {1, 1};
  int dummy_outputs_count[1] = {1};

  if (world.size() < 5 || world.rank() >= 4) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dummy_inputs_count[0]));
    taskData->inputs_count.emplace_back(1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dummy_inputs_count[1]));
    taskData->inputs_count.emplace_back(1);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(&dummy_outputs_count[0]));
    taskData->outputs_count.emplace_back(1);
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator calculator(taskData);
  ASSERT_TRUE(calculator.validation());
}

TEST(golovkin_integration_rectangular_method, test_post_processing) {
  boost::mpi::communicator world;
  double lower_limit = 0.0;
  double upper_limit = 10.0;
  int partitions = 1000;

  std::vector<double> computed_result(1, 0.0);
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.size() < 5 || world.rank() >= 4) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskData->inputs_count.emplace_back(1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskData->inputs_count.emplace_back(1);
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partitions));
    taskData->inputs_count.emplace_back(1);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskData->outputs_count.emplace_back(computed_result.size());
  }

  golovkin_integration_rectangular_method::MPIIntegralCalculator calculator(taskData);
  calculator.set_function([](double x) { return 5.0; });

  calculator.validation();
  calculator.pre_processing();
  calculator.run();
  calculator.post_processing();

  if (world.size() < 5 || world.rank() >= 4) {
    ASSERT_NEAR(computed_result[0], 50.0, 1e-3);
  }
}

TEST(golovkin_integration_rectangular_method, test_constant_function_validation) {
  boost::mpi::communicator world;
  std::vector<double> computed_result(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  double lower_limit = 0.0;
  double upper_limit = 10.0;
  int partition_count = 1000000;

  if (world.size() < 5 || world.rank() >= 4) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_limit));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&partition_count));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(computed_result.data()));
    taskDataPar->outputs_count.emplace_back(computed_result.size());

    golovkin_integration_rectangular_method::MPIIntegralCalculator valid_task(taskDataPar);
    ASSERT_EQ(valid_task.validation(), true);

    taskDataPar->inputs_count[1] = 0;
    golovkin_integration_rectangular_method::MPIIntegralCalculator invalid_task_missing_input(taskDataPar);
    ASSERT_EQ(invalid_task_missing_input.validation(), false);

    taskDataPar->inputs_count[1] = 1;
    taskDataPar->outputs_count[0] = 0;
    golovkin_integration_rectangular_method::MPIIntegralCalculator invalid_task_missing_output(taskDataPar);
    ASSERT_EQ(invalid_task_missing_output.validation(), false);
  }
}