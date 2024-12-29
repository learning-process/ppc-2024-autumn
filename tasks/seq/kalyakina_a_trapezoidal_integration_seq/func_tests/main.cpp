// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "seq/kalyakina_a_trapezoidal_integration_seq/include/ops_seq.hpp"

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
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(count.data()));
  taskDataSequential->inputs_count.emplace_back(count.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
  taskDataSequential->inputs_count.emplace_back(limits.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
  taskDataSequential->inputs_count.emplace_back(intervals.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSequential->outputs_count.emplace_back(out.size());

  kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask TaskSequential(taskDataSequential, function);

  ASSERT_EQ(TaskSequential.validation(), false);
}

void TestOfFunction(double (*function)(std::vector<double>), std::vector<unsigned int>& count,
                    std::vector<std::pair<double, double>>& limits, std::vector<unsigned int>& intervals,
                    double answer) {
  std::vector<double> out(1, 0.0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSequential = std::make_shared<ppc::core::TaskData>();

  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(count.data()));
  taskDataSequential->inputs_count.emplace_back(count.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(limits.data()));
  taskDataSequential->inputs_count.emplace_back(limits.size());
  taskDataSequential->inputs.emplace_back(reinterpret_cast<uint8_t*>(intervals.data()));
  taskDataSequential->inputs_count.emplace_back(intervals.size());
  taskDataSequential->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSequential->outputs_count.emplace_back(out.size());

  kalyakina_a_trapezoidal_integration_seq::TrapezoidalIntegrationTask TaskSequential(taskDataSequential, function);

  ASSERT_EQ(TaskSequential.validation(), true);
  TaskSequential.pre_processing();
  TaskSequential.run();
  TaskSequential.post_processing();

  EXPECT_NEAR(answer, out[0], 0.05);
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_validation_count_of_variables) {
  std::vector<unsigned int> count(1, 3);
  std::vector<std::pair<double, double>> limits = {{2.5, 4.5}, {1.0, 3.2}};
  std::vector<unsigned int> intervals = {100, 100};

  TestOfValidation(function1, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_validation_size_numbers_of_intervals) {
  std::vector<unsigned int> count(1, 2);
  std::vector<std::pair<double, double>> limits = {{2.5, 4.5}, {1.0, 3.2}};
  std::vector<unsigned int> intervals = {100};

  TestOfValidation(function1, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_validation_size_limits) {
  std::vector<unsigned int> count(1, 2);
  std::vector<std::pair<double, double>> limits = {{2.5, 4.5}};
  std::vector<unsigned int> intervals = {100, 100};

  TestOfValidation(function1, count, limits, intervals);
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_functionality_1) {
  std::vector<unsigned int> count(1, 2);
  std::vector<std::pair<double, double>> limits = {{2.5, 4.5}, {1.0, 3.2}};
  std::vector<unsigned int> intervals = {100, 100};

  TestOfFunction(function1, count, limits, intervals,
                 (3.2 - 1) * (pow(4.5, 4) - pow(2.5, 4)) / 4 + (4.5 - 2.5) * (pow(3.2, 4) - pow(1, 4)) / 4);
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_functionality_2) {
  std::vector<unsigned int> count(1, 3);
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}, {0.0, 1.0}};
  std::vector<unsigned int> intervals = {50, 50, 50};

  TestOfFunction(function2, count, limits, intervals, -3 * std::cos(1) + 3 * std::cos(0));
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_functionality_3) {
  std::vector<unsigned int> count(1, 3);
  std::vector<std::pair<double, double>> limits = {{0.0, 3.0}, {0.0, 4.0}, {0.0, 5.0}};
  std::vector<unsigned int> intervals = {50, 50, 50};

  TestOfFunction(function3, count, limits, intervals, pow(3 - 0, 2) * pow(4 - 0, 2) * pow(5 - 0, 2));
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_functionality_4) {
  std::vector<unsigned int> count(1, 1);
  std::vector<std::pair<double, double>> limits = {{0.0, 0.5}};
  std::vector<unsigned int> intervals = {1000};

  TestOfFunction(function4, count, limits, intervals, acos(0.5) - acos(0));
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_functionality_5) {
  std::vector<unsigned int> count(1, 2);
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {0.0, 1.0}};
  std::vector<unsigned int> intervals = {100, 100};

  TestOfFunction(function5, count, limits, intervals, (sin(1) - sin(0)) * (cos(1) - cos(0)));
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_functionality_6) {
  std::vector<unsigned int> count(1, 2);
  std::vector<std::pair<double, double>> limits = {{0.0, 1.0}, {4.0, 6.0}};
  std::vector<unsigned int> intervals = {100, 100};

  TestOfFunction(function6, count, limits, intervals, (pow(6.0, 3) - pow(4.0, 3)) * (cos(5 * 1.0) - cos(5 * 0.0)) / 10);
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_functionality_random_limits) {
  std::vector<unsigned int> count(1, 3);
  std::vector<std::pair<double, double>> limits = {GetRandomLimit(0.0, 1.0), GetRandomLimit(0.0, 1.0),
                                                   GetRandomLimit(0.0, 1.0)};
  std::vector<unsigned int> intervals = {50, 50, 50};

  TestOfFunction(function3, count, limits, intervals,
                 (pow(limits[0].second, 2) - pow(limits[0].first, 2)) *
                     (pow(limits[1].second, 2) - pow(limits[1].first, 2)) *
                     (pow(limits[2].second, 2) - pow(limits[2].first, 2)));
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_functionality_intervals) {
  std::vector<unsigned int> count(1, 3);
  std::vector<std::pair<double, double>> limits = {{0.0, 3.0}, {0.0, 4.0}, {0.0, 5.0}};
  std::vector<unsigned int> intervals = {GetRandomIntegerData(40, 50), GetRandomIntegerData(40, 50),
                                         GetRandomIntegerData(40, 50)};

  TestOfFunction(function3, count, limits, intervals, pow(3 - 0, 2) * pow(4 - 0, 2) * pow(5 - 0, 2));
}

TEST(kalyakina_a_trapezoidal_integration_seq, Test_of_functionality_limits_and_intervals) {
  std::vector<unsigned int> count(1, 3);
  std::vector<std::pair<double, double>> limits = {GetRandomLimit(0.0, 1.0), GetRandomLimit(0.0, 1.0),
                                                   GetRandomLimit(0.0, 1.0)};
  std::vector<unsigned int> intervals = {GetRandomIntegerData(40, 50), GetRandomIntegerData(40, 50),
                                         GetRandomIntegerData(40, 50)};

  TestOfFunction(function3, count, limits, intervals,
                 (pow(limits[0].second, 2) - pow(limits[0].first, 2)) *
                     (pow(limits[1].second, 2) - pow(limits[1].first, 2)) *
                     (pow(limits[2].second, 2) - pow(limits[2].first, 2)));
}