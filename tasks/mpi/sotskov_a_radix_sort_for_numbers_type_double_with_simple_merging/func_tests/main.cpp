#include <gtest/gtest.h>
#include <mpi.h>

#include <random>

#include "core/task/include/task.hpp"
#include "mpi/sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging/include/ops_mpi.hpp"

namespace sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi {
std::vector<double> generateRandomData(int size, double minValue, double maxValue) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(minValue, maxValue);

  std::vector<double> data(size);
  for (int i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }

  return data;
}

void runSortingTest(const std::vector<double>& testData) {
  int rank;
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dataSize = testData.size();
  std::vector<double> parallelResult(dataSize, 0.0);
  std::vector<double> sequentialResult(dataSize, 0.0);

  auto parallelTaskData = std::make_shared<ppc::core::TaskData>();
  auto sequentialTaskData = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(const_cast<double*>(testData.data())));
    parallelTaskData->inputs_count.emplace_back(dataSize);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    parallelTaskData->outputs_count.emplace_back(dataSize);

    sequentialTaskData->inputs = parallelTaskData->inputs;
    sequentialTaskData->inputs_count = parallelTaskData->inputs_count;
    sequentialTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequentialResult.data()));
    sequentialTaskData->outputs_count.emplace_back(dataSize);
  }

  sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskParallel parallelRadixSort(
      parallelTaskData);
  ASSERT_TRUE(parallelRadixSort.validation());
  parallelRadixSort.pre_processing();
  parallelRadixSort.run();
  parallelRadixSort.post_processing();

  if (rank == 0) {
    sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::TestMPITaskSequential sequentialRadixSort(
        sequentialTaskData);
    ASSERT_TRUE(sequentialRadixSort.validation());
    sequentialRadixSort.pre_processing();
    sequentialRadixSort.run();
    sequentialRadixSort.post_processing();

    for (int i = 0; i < dataSize; ++i) {
      ASSERT_NEAR(parallelResult[i], sequentialResult[i], 1e-10);
    }
  }
}
}  // namespace sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, VerifySorting) {
  const int dataSize = 100;
  const double minValue = 0.0;
  const double maxValue = 100.0;
  std::vector<double> testData =
      sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::generateRandomData(dataSize, minValue,
                                                                                               maxValue);
  sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::runSortingTest(testData);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, VerifySortingWithPreGeneratedData) {
  std::vector<double> testData = {10.5, 2.3, 4.7, 8.0, 1.2, 3.5, 7.8, 6.1, 5.1};
  sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::runSortingTest(testData);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, VerifySortingEmptyData) {
  std::vector<double> testData = {};
  sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::runSortingTest(testData);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, VerifySortingSingleElement) {
  std::vector<double> testData = {42.0};
  sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::runSortingTest(testData);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, VerifySortingLargeData) {
  const int dataSize = 1000;
  const double minValue = 0.0;
  const double maxValue = 100.0;
  std::vector<double> testData =
      sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::generateRandomData(dataSize, minValue,
                                                                                               maxValue);
  sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::runSortingTest(testData);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, VerifySortingAlreadySortedData) {
  std::vector<double> testData = {1.0, 2.0, 3.0, 4.0, 5.0};
  sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::runSortingTest(testData);
}

TEST(MPIEnvironment, VerifySortingWithNegativeNumbers) {
  std::vector<double> testData = {-10.5, -2.3, -4.7, -8.0, -1.2, -3.5, -7.8, -6.1, -5.0};
  sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::runSortingTest(testData);
}

TEST(sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi, VerifySortingWithMixedRangeValues) {
  std::vector<double> testData = {10000.0, 0.0001, 50.0, 999.99, 1.0, 1000.0};
  sotskov_a_radix_sort_for_numbers_type_double_with_simple_merging_mpi::runSortingTest(testData);
}
