#include <gtest/gtest.h>

#include <memory>
#include <random>
#include <vector>

#include "seq/korneeva_e_num_of_orderly_violations/include/ops_seq.hpp"

// Function to create a test environment
template <typename T>
std::shared_ptr<ppc::core::TaskData> createTaskData(const std::vector<T>& data) {
  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs_count.push_back(data.size());
  taskData->inputs.push_back(const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(data.data())));
  return taskData;
}

// General function for preparing data and checking results
template <typename T>
void runOrderlyViolationsTest(const std::vector<T>& data, int expectedViolations) {
  auto taskData = createTaskData(data);
  korneeva_e_num_of_orderly_violations_seq::OrderlyViolationsCounter<T, int> counter(taskData);
  ASSERT_EQ(counter.count_orderly_violations(data), expectedViolations);
}

// Function for tests with random numbers
template <typename T>
void runRandomTest(int size) {
  std::vector<T> numbers(size);
  std::random_device randomDevice;
  std::default_random_engine randomEngine(randomDevice());
  std::uniform_int_distribution<int> distribution(0, size);

  std::generate(numbers.begin(), numbers.end(), [&distribution, &randomEngine] { return distribution(randomEngine); });

  auto taskDataPtr = std::make_shared<ppc::core::TaskData>();
  taskDataPtr->inputs.emplace_back(reinterpret_cast<uint8_t*>(numbers.data()));
  taskDataPtr->inputs_count.emplace_back(numbers.size());

  std::vector<int> output(1);
  taskDataPtr->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  taskDataPtr->outputs_count.emplace_back(1);

  korneeva_e_num_of_orderly_violations_seq::OrderlyViolationsCounter<T, int> violationCounter(taskDataPtr);

  ASSERT_EQ(violationCounter.validation(), true);

  violationCounter.pre_processing();
  violationCounter.run();
  violationCounter.post_processing();

  int result = violationCounter.count_orderly_violations(numbers);
  ASSERT_EQ(result, output[0]);
}

TEST(korneeva_e_num_of_orderly_violations_seq, NoViolations) { runOrderlyViolationsTest<int>({1, 2, 3, 4, 5}, 0); }

TEST(korneeva_e_num_of_orderly_violations_seq, AllViolations) { runOrderlyViolationsTest<int>({5, 4, 3, 2, 1}, 4); }

TEST(korneeva_e_num_of_orderly_violations_seq, MixedViolations) { runOrderlyViolationsTest<int>({1, 3, 2, 4, 0}, 2); }

TEST(korneeva_e_num_of_orderly_violations_seq, SingleElement) { runOrderlyViolationsTest<int>({42}, 0); }

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Negative_Elements) {
  runOrderlyViolationsTest<int>({-1, -2, -3, -4, -5}, 4);
}

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Repeating_Elements) {
  runOrderlyViolationsTest<int>({1, 2, 2, 1, 2}, 1);
}

TEST(korneeva_e_num_of_orderly_violations_seq, Test_0_int) { runOrderlyViolationsTest<int>({}, 0); }

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Random_10_int) { runRandomTest<int>(10); }

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Random_100_int) { runRandomTest<int>(100); }

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Random_1000_int) { runRandomTest<int>(1000); }

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Random_10000_int) { runRandomTest<int>(10000); }

TEST(korneeva_e_num_of_orderly_violations_seq, NoViolations_double) {
  runOrderlyViolationsTest<double>({1.1, 2.2, 3.3, 4.4, 5.5}, 0);
}

TEST(korneeva_e_num_of_orderly_violations_seq, AllViolations_double) {
  runOrderlyViolationsTest<double>({5.5, 4.4, 3.3, 2.2, 1.1}, 4);
}

TEST(korneeva_e_num_of_orderly_violations_seq, MixedViolations_double) {
  runOrderlyViolationsTest<double>({1.1, 3.3, 2.2, 4.4, 0.0}, 2);
}

TEST(korneeva_e_num_of_orderly_violations_seq, SingleElement_double) { runOrderlyViolationsTest<double>({42.0}, 0); }

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Negative_Elements_double) {
  runOrderlyViolationsTest<double>({-1.1, -2.2, -3.3, -4.4, -5.5}, 4);
}

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Repeating_Elements_double) {
  runOrderlyViolationsTest<double>({1.1, 2.2, 2.2, 1.1, 2.2}, 1);
}

TEST(korneeva_e_num_of_orderly_violations_seq, Test_0_double) { runOrderlyViolationsTest<double>({}, 0); }

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Random_10_double) { runRandomTest<double>(10); }

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Random_100_double) { runRandomTest<double>(100); }

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Random_1000_double) { runRandomTest<double>(1000); }

TEST(korneeva_e_num_of_orderly_violations_seq, Test_Random_10000_double) { runRandomTest<double>(10000); }
