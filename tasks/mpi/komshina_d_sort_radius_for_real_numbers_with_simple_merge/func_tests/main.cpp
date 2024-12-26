#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace mpi = boost::mpi;
using namespace komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi;

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, TestParallelSorting) {
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> parallelTaskData = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> sequentialTaskData = std::make_shared<ppc::core::TaskData>();

  int dataSize = 8;
  std::vector<double> inputValues = {10.1, 8.1, 0.2, 1.5, -6.3, 4.4, -11.4, 0.6};
  std::vector<double> parallelResult(dataSize, 0.0);
  std::vector<double> sequentialResult(dataSize, 0.0);

  if (world.rank() == 0) {
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dataSize));
    parallelTaskData->inputs_count.emplace_back(1);
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputValues.data()));
    parallelTaskData->inputs_count.emplace_back(dataSize);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    parallelTaskData->outputs_count.emplace_back(dataSize);

    sequentialTaskData->inputs = parallelTaskData->inputs;
    sequentialTaskData->inputs_count = parallelTaskData->inputs_count;
    sequentialTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequentialResult.data()));
    sequentialTaskData->outputs_count.emplace_back(dataSize);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(parallelTaskData);
  ASSERT_TRUE(parallelSortTask.validation());
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential sequentialSortTask(
        sequentialTaskData);
    ASSERT_TRUE(sequentialSortTask.validation());
    sequentialSortTask.pre_processing();
    sequentialSortTask.run();
    sequentialSortTask.post_processing();

    auto* parallelResults = reinterpret_cast<double*>(parallelTaskData->outputs[0]);
    auto* sequentialResults = reinterpret_cast<double*>(sequentialTaskData->outputs[0]);

    for (int i = 0; i < dataSize; ++i) {
      ASSERT_NEAR(parallelResults[i], sequentialResults[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, TestEmptyInput) {
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> parallelTaskData = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> sequentialTaskData = std::make_shared<ppc::core::TaskData>();

  int dataSize = 0;
  std::vector<double> inputValues;
  std::vector<double> parallelResult(dataSize, 0.0);
  std::vector<double> sequentialResult(dataSize, 0.0);

  if (world.rank() == 0) {
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dataSize));
    parallelTaskData->inputs_count.emplace_back(1);
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputValues.data()));
    parallelTaskData->inputs_count.emplace_back(dataSize);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    parallelTaskData->outputs_count.emplace_back(dataSize);

    sequentialTaskData->inputs = parallelTaskData->inputs;
    sequentialTaskData->inputs_count = parallelTaskData->inputs_count;
    sequentialTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequentialResult.data()));
    sequentialTaskData->outputs_count.emplace_back(dataSize);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(parallelTaskData);
  ASSERT_TRUE(parallelSortTask.validation());
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential sequentialSortTask(
        sequentialTaskData);
    ASSERT_TRUE(sequentialSortTask.validation());
    sequentialSortTask.pre_processing();
    sequentialSortTask.run();
    sequentialSortTask.post_processing();

    auto* parallelResults = reinterpret_cast<double*>(parallelTaskData->outputs[0]);
    auto* sequentialResults = reinterpret_cast<double*>(sequentialTaskData->outputs[0]);

    for (int i = 0; i < dataSize; ++i) {
      ASSERT_NEAR(parallelResults[i], sequentialResults[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, TestSingleElementInput) {
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> parallelTaskData = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> sequentialTaskData = std::make_shared<ppc::core::TaskData>();

  int dataSize = 1;
  std::vector<double> inputValues = {42.0};
  std::vector<double> parallelResult(dataSize, 0.0);
  std::vector<double> sequentialResult(dataSize, 0.0);

  if (world.rank() == 0) {
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dataSize));
    parallelTaskData->inputs_count.emplace_back(1);
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputValues.data()));
    parallelTaskData->inputs_count.emplace_back(dataSize);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    parallelTaskData->outputs_count.emplace_back(dataSize);

    sequentialTaskData->inputs = parallelTaskData->inputs;
    sequentialTaskData->inputs_count = parallelTaskData->inputs_count;
    sequentialTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequentialResult.data()));
    sequentialTaskData->outputs_count.emplace_back(dataSize);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(parallelTaskData);
  ASSERT_TRUE(parallelSortTask.validation());
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential sequentialSortTask(
        sequentialTaskData);
    ASSERT_TRUE(sequentialSortTask.validation());
    sequentialSortTask.pre_processing();
    sequentialSortTask.run();
    sequentialSortTask.post_processing();

    auto* parallelResults = reinterpret_cast<double*>(parallelTaskData->outputs[0]);
    auto* sequentialResults = reinterpret_cast<double*>(sequentialTaskData->outputs[0]);

    ASSERT_NEAR(parallelResults[0], sequentialResults[0], 1e-12);
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, TestLargeInput) {
  mpi::communicator world;

  std::shared_ptr<ppc::core::TaskData> parallelTaskData = std::make_shared<ppc::core::TaskData>();
  std::shared_ptr<ppc::core::TaskData> sequentialTaskData = std::make_shared<ppc::core::TaskData>();

  int dataSize = 100000;
  std::vector<double> inputValues(dataSize);
  std::generate(inputValues.begin(), inputValues.end(), std::rand);
  std::vector<double> parallelResult(dataSize, 0.0);
  std::vector<double> sequentialResult(dataSize, 0.0);

  if (world.rank() == 0) {
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(&dataSize));
    parallelTaskData->inputs_count.emplace_back(1);
    parallelTaskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputValues.data()));
    parallelTaskData->inputs_count.emplace_back(dataSize);
    parallelTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallelResult.data()));
    parallelTaskData->outputs_count.emplace_back(dataSize);

    sequentialTaskData->inputs = parallelTaskData->inputs;
    sequentialTaskData->inputs_count = parallelTaskData->inputs_count;
    sequentialTaskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequentialResult.data()));
    sequentialTaskData->outputs_count.emplace_back(dataSize);
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(parallelTaskData);
  ASSERT_TRUE(parallelSortTask.validation());
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestTaskSequential sequentialSortTask(
        sequentialTaskData);
    ASSERT_TRUE(sequentialSortTask.validation());
    sequentialSortTask.pre_processing();
    sequentialSortTask.run();
    sequentialSortTask.post_processing();

    auto* parallelResults = reinterpret_cast<double*>(parallelTaskData->outputs[0]);
    auto* sequentialResults = reinterpret_cast<double*>(sequentialTaskData->outputs[0]);

    for (int i = 0; i < dataSize; ++i) {
      ASSERT_NEAR(parallelResults[i], sequentialResults[i], 1e-12);
    }
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, TestDoubleToUint64Conversion) {
  std::vector<double> inputValues = {10.1, -6.3, 4.4, 0.6};

  std::vector<uint64_t> keys(inputValues.size(), 0);

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::convert_doubles_to_uint64(inputValues, keys);

  for (size_t i = 0; i < inputValues.size(); ++i) {
    uint64_t expectedKey;
    std::memcpy(&expectedKey, &inputValues[i], sizeof(double));

    expectedKey = ((expectedKey >> 63) & 1) != 0 ? ~expectedKey : (expectedKey | (1ULL << 63));

    ASSERT_EQ(keys[i], expectedKey) << "Index: " << i << " Expected: " << expectedKey << " Found: " << keys[i];
  }
}
