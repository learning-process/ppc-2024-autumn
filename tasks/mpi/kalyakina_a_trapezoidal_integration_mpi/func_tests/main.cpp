// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <vector>

#include "mpi/kalyakina_a_trapezoidal_integration_mpi/include/ops_mpi.hpp"

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
                    std::vector<std::pair<double, double>>& limits, std::vector<unsigned int>& intervals,
                    double answer) {
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
  ASSERT_EQ(TaskParallel.validation(), true);
  TaskParallel.pre_processing();
  TaskParallel.run();
  TaskParallel.post_processing();

  if (world.rank() == 0) {
    EXPECT_NEAR(answer, out[0], 0.001);
  }
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_validation_count_of_variables) {
  boost::mpi::communicator world;

  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;
  std::vector<unsigned int> count;

  if (world.rank() == 0) {
    limits = {{2.5, 4.5}, {1.0, 3.2}};
    intervals = {1000, 1000};
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
    intervals = {1000, 1000};
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
    intervals = {1000, 1000};
    count = std::vector<unsigned int>{2};
  }

  TestOfFunction(function1, count, limits, intervals,
                 (3.2 - 1) * (pow(4.5, 4) - pow(2.5, 4)) / 4 + (4.5 - 2.5) * (pow(3.2, 4) - pow(1, 4)) / 4);
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_2) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{3};
    limits = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
    intervals = {100, 100, 100};
  }

  TestOfFunction(function2, count, limits, intervals, -3 * std::cos(1) + 3 * std::cos(0));
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_3) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{3};
    limits = {{0.0, 3.0}, {0.0, 4.0}, {0.0, 5.0}};
    intervals = {100, 100, 100};
  }

  TestOfFunction(function3, count, limits, intervals, pow(3 - 0, 2) * pow(4 - 0, 2) * pow(5 - 0, 2));
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_4) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{1};
    limits = {{0.0, 0.5}};
    intervals = {10000};
  }

  TestOfFunction(function4, count, limits, intervals, acos(0.5) - acos(0));
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_5) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{2};
    limits = {{0.0, 1.0}, {0.0, 1.0}};
    intervals = {1000, 1000};
  }

  TestOfFunction(function5, count, limits, intervals, (sin(1) - sin(0)) * (cos(1) - cos(0)));
}

TEST(kalyakina_a_trapezoidal_integration_mpi, Test_of_functionality_6) {
  boost::mpi::communicator world;

  std::vector<unsigned int> count;
  std::vector<std::pair<double, double>> limits;
  std::vector<unsigned int> intervals;

  if (world.rank() == 0) {
    count = std::vector<unsigned int>{2};
    limits = {{0.0, 1.0}, {4.0, 6.0}};
    intervals = {1000, 1000};
  }

  TestOfFunction(function6, count, limits, intervals, (pow(6.0, 3) - pow(4.0, 3)) * (cos(5 * 1.0) - cos(5 * 0.0)) / 10);
}