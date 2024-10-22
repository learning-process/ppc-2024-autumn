// Copyright 2024 Lupsha Egor
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/lupsha_e_rect_integration/include/ops_mpi.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TEST(lupsha_e_rect_integration_mpi, Test_Constant) {
  boost::mpi::communicator world;
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  int num_intervals = 1000;
  std::vector<double> global_sum(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  lupsha_e_rect_integration_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  std::function<double(double)> f = [](double x) { return 10.0; };
  testMpiTaskParallel.function_set(f);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataSeq->inputs_count.emplace_back(1);

    lupsha_e_rect_integration_mpi::TestMPITaskSequential sequential_Task(taskDataSeq);
    sequential_Task.function_set(f);
    double expected_result = 10.0;
    ASSERT_NEAR(global_sum[0], expected_result, 1e-5);
  }
}

TEST(lupsha_e_rect_integration_mpi, Test_Logarithm) {
  boost::mpi::communicator world;
  double lower_bound = 0.1;
  double upper_bound = 1.0;
  int num_intervals = 10000;
  std::vector<double> global_sum(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  lupsha_e_rect_integration_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  std::function<double(double)> f = [](double x) { return std::log(x); };
  testMpiTaskParallel.function_set(f);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataSeq->inputs_count.emplace_back(1);

    lupsha_e_rect_integration_mpi::TestMPITaskSequential sequential_Task(taskDataSeq);
    sequential_Task.function_set(f);
    double expected_result = 
        (upper_bound * log(upper_bound) - upper_bound) - (lower_bound * log(lower_bound) - lower_bound);
    ASSERT_NEAR(global_sum[0], expected_result, 1e-3);
  }
}

TEST(lupsha_e_rect_integration_mpi, Test_Gaussian) {
  boost::mpi::communicator world;
  double lower_bound = -1.0;
  double upper_bound = 1.0;
  int num_intervals = 1000;
  std::vector<double> global_sum(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  lupsha_e_rect_integration_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  std::function<double(double)> f = [](double x) { return std::exp(-x * x); };
  testMpiTaskParallel.function_set(f);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataSeq->inputs_count.emplace_back(1);

    lupsha_e_rect_integration_mpi::TestMPITaskSequential sequential_Task(taskDataSeq);
    sequential_Task.function_set(f);

    double expected_result = std::sqrt(M_PI) * (std::erf(upper_bound) - std::erf(lower_bound)) / 2;
    ASSERT_NEAR(global_sum[0], expected_result, 1e-5);
  }
}

TEST(lupsha_e_rect_integration_mpi, Test_Cos) {
  boost::mpi::communicator world;
  double lower_bound = 0.0;
  double upper_bound = 2 * M_PI;
  int num_intervals = 1000;
  std::vector<double> global_sum(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  lupsha_e_rect_integration_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  std::function<double(double)> f = [](double x) { return cos(x); };
  testMpiTaskParallel.function_set(f);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataSeq->inputs_count.emplace_back(1);

    lupsha_e_rect_integration_mpi::TestMPITaskSequential sequential_Task(taskDataSeq);
    sequential_Task.function_set(f);
    double expected_result = sin(upper_bound) - sin(lower_bound);
    ASSERT_NEAR(global_sum[0], expected_result, 1e-5);
  }
}

TEST(lupsha_e_rect_integration_mpi, Test_Power) {
  boost::mpi::communicator world;
  double lower_bound = 0.0;
  double upper_bound = 1.0;
  int num_intervals = 1000;
  std::vector<double> global_sum(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  lupsha_e_rect_integration_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  std::function<double(double)> f = [](double x) { return x * x; };
  testMpiTaskParallel.function_set(f);

  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&lower_bound));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&upper_bound));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&num_intervals));
    taskDataSeq->inputs_count.emplace_back(1);

    lupsha_e_rect_integration_mpi::TestMPITaskSequential sequential_Task(taskDataSeq);
    sequential_Task.function_set(f);
    double expected_result = 
        (upper_bound * upper_bound * upper_bound / 3) - (lower_bound * lower_bound * lower_bound / 3);
    ASSERT_NEAR(global_sum[0], expected_result, 1e-3);
  }
}

int main(int argc, char** argv) {
  boost::mpi::environment env(argc, argv);
  boost::mpi::communicator world;
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (world.rank() != 0) {
    delete listeners.Release(listeners.default_result_printer());
  }
  return RUN_ALL_TESTS();
}
