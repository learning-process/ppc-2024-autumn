// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "mpi/kalyakina_a_trapezoidal_integration_mpi/include/ops_mpi.hpp"

std::pair<double, double> GetRandomLimit(double min_value, double max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::pair<double, double> result;
  max_value = max_value * 1000 - 5;
  min_value *= 1000;
  result.first = (double)(gen() % (int)(max_value - min_value) + min_value) / 1000;
  max_value += 5;
  min_value = result.first * 1000;
  result.second = (double)(gen() % (int)(max_value - min_value) + min_value) / 1000;
  return result;
}

unsigned int GetRandomIntegerData(unsigned int min_value, unsigned int max_value) {
  std::random_device dev;
  std::mt19937 gen(dev());
  return gen() % (max_value - min_value) + min_value;
}

double function1(std::vector<double> input) { return pow(input[0], 3) + pow(input[1], 3); };
double function2(std::vector<double> input) { return sin(input[0]) + sin(input[1]) + sin(input[2]); };
double function3(std::vector<double> input) { return 8 * input[0] * input[1] * input[2]; };
double function4(std::vector<double> input) { return -1.0 / sqrt(1 - pow(input[0], 2)); };
double function5(std::vector<double> input) { return -(sin(input[0]) * cos(input[1])); };
double function6(std::vector<double> input) { return (-3 * pow(input[1], 2) * sin(5 * input[0])) / 2; };

void TestOfValidation(double (*function)(std::vector<double>), std::vector<unsigned int>& count,
                      std::vector<std::pair<double, double>>& limits, std::vector<unsigned int>& intervals) {
  boost::mpi::communicator world;

  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  // Create TaskData

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(count.data()));
    taskDataParallel->inputs_count.emplace_back(count.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
    taskDataParallel->inputs_count.emplace_back(limits.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataParallel->inputs_count.emplace_back(intervals.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataParallel->outputs_count.emplace_back(out.size());
  }

  kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel TaskParallel(taskDataParallel, function);
  if (world.rank() == 0) {
    ASSERT_EQ(TaskParallel.validation(), false);
  }
}

void TestOfFunction(double (*function)(std::vector<double>), std::vector<unsigned int>& count,
                    std::vector<std::pair<double, double>>& limits, std::vector<unsigned int>& intervals) {
  boost::mpi::communicator world;

  std::vector<double> out_mpi(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataParallel = std::make_shared<ppc::core::TaskData>();

  // Create TaskData

  if (world.rank() == 0) {
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(count.data()));
    taskDataParallel->inputs_count.emplace_back(count.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
    taskDataParallel->inputs_count.emplace_back(limits.size());
    taskDataParallel->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataParallel->inputs_count.emplace_back(intervals.size());
    taskDataParallel->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_mpi.data()));
    taskDataParallel->outputs_count.emplace_back(out_mpi.size());
  }

  kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskParallel TaskParallel(taskDataParallel, function);
  ASSERT_EQ(TaskParallel.validation(), true);
  TaskParallel.pre_processing();
  TaskParallel.run();
  TaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<double> out_seq(1, 0.0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(count.data()));
    taskDataSequential->inputs_count.emplace_back(count.size());
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
    taskDataSequential->inputs_count.emplace_back(limits.size());
    taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
    taskDataSequential->inputs_count.emplace_back(intervals.size());
    taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskDataSequential->outputs_count.emplace_back(out_seq.size());

    // Create Task
    kalyakina_a_trapezoidal_integration_mpi::TrapezoidalIntegrationTaskSequential TaskSequential(taskDataSequential,
                                                                                                 function);

    ASSERT_EQ(TaskSequential.validation(), true);
    TaskSequential.pre_processing();
    TaskSequential.run();
    TaskSequential.post_processing();

    EXPECT_NEAR(out_seq[0], out_mpi[0], 0.00001);
  }
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_validation_count_of_variables) {
  boost::mpi::communicator world;

  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;
  std::vector<unsigned int> count;

  if (world.rank() == 0) {
    limits = {{2.5, 4.5}, {1.0, 3.2}};
    intervals = {100, 100};
    count = std::vector<unsigned int>{3};
  }

  TestOfValidation(function1, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_validation_size_numbers_of_intervals) {
  boost::mpi::communicator world;

  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;
  std::vector<unsigned int> count;

  if (world.rank() == 0) {
    limits = {{2.5, 4.5}, {1.0, 3.2}};
    intervals = {1000};
    count = std::vector<unsigned int>{2};
  }

  TestOfValidation(function1, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_validation_size_limits) {
  boost::mpi::communicator world;

  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;
  std::vector<unsigned int> count;

  if (world.rank() == 0) {
    limits = {{2.5, 4.5}};
    intervals = {100, 100};
    count = std::vector<unsigned int>{2};
  }

  TestOfValidation(function1, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_1) {
  boost::mpi::communicator world;

  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;
  std::vector<unsigned int> count;

  if (world.rank() == 0) {
    limits = {{2.5, 4.5}, {1.0, 3.2}};
    intervals = {100, 100};
    count = std::vector<unsigned int>{2};
  }

  TestOfFunction(function1, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_2) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{3};
    limits = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
    intervals = {80, 80, 80};
  }

  TestOfFunction(function2, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_3) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{3};
    limits = {{0.0, 3.0}, {0.0, 4.0}, {0.0, 5.0}};
    intervals = {80, 80, 80};
  }

  TestOfFunction(function3, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_4) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{1};
    limits = {{0.0, 0.5}};
    intervals = {1000};
  }

  TestOfFunction(function4, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_5) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{2};
    limits = {{0.0, 1.0}, {0.0, 1.0}};
    intervals = {100, 100};
  }

  TestOfFunction(function5, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_6) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{2};
    limits = {{0.0, 1.0}, {4.0, 6.0}};
    intervals = {100, 100};
  }

  TestOfFunction(function6, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_random_limits) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{3};
    limits = {GetRandomLimit(0.0, 10.0), GetRandomLimit(0.0, 10.0), GetRandomLimit(0.0, 10.0)};
    intervals = {80, 80, 80};
  }

  TestOfFunction(function3, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_random_intervals) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{2};
    limits = {{0.0, 1.0}, {4.0, 6.0}};
    intervals = {GetRandomIntegerData(100, 150), GetRandomIntegerData(100, 150)};
  }

  TestOfFunction(function6, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_random_limits_and_intervals) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{3};
    limits = {GetRandomLimit(0.0, 10.0), GetRandomLimit(0.0, 10.0), GetRandomLimit(0.0, 10.0)};
    intervals = {GetRandomIntegerData(40, 60), GetRandomIntegerData(40, 60), GetRandomIntegerData(40, 60)};
  }

  TestOfFunction(function3, count, limits, intervals);
}