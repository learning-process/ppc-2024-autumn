#include <gtest/gtest.h>

#include "mpi/chastov_v_bubble_sort/include/ops_mpi.hpp"

TEST(chastov_v_bubble_sort, zero_len) {
  std::vector<int> input_data;
  std::vector<int> output_data;
  boost::mpi::communicator communicator;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (communicator.rank() == 0) {
    taskDataPar->inputs_count.push_back(static_cast<int>(input_data.size()));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.push_back(static_cast<int>(output_data.size()));
  }

  chastov_v_bubble_sort::TestMPITaskParallel<int> parallel_task(taskDataPar);

  if (communicator.rank() == 0) {
    EXPECT_EQ(false, parallel_task.validation());
  }
}

TEST(chastov_v_bubble_sort, sorted_input) {
  std::vector<int> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<int> output_data(input_data.size(), 0);
  boost::mpi::communicator communicator;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (communicator.rank() == 0) {
    taskDataPar->inputs_count.push_back(static_cast<int>(input_data.size()));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.push_back(static_cast<int>(output_data.size()));
  }

  chastov_v_bubble_sort::TestMPITaskParallel<int> parallel_task(taskDataPar);

  ASSERT_TRUE(parallel_task.validation());
  ASSERT_TRUE(parallel_task.pre_processing());
  ASSERT_TRUE(parallel_task.run());
  ASSERT_TRUE(parallel_task.post_processing());

  if (communicator.rank() == 0) {
    EXPECT_EQ(input_data, output_data);
  }
}

TEST(chastov_v_bubble_sort, invalid_input) {
  std::vector<int> input_data;
  std::vector<int> output_data(10);
  boost::mpi::communicator communicator;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (communicator.rank() == 0) {
    taskDataPar->inputs_count.push_back(static_cast<int>(input_data.size()));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.push_back(static_cast<int>(output_data.size()));
  }

  chastov_v_bubble_sort::TestMPITaskParallel<int> parallel_task(taskDataPar);

  if (communicator.rank() == 0) {
    EXPECT_FALSE(parallel_task.validation());
  }
}

TEST(chastov_v_bubble_sort, reverse_sorted_input) {
  std::vector<int> input_data = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> output_data(input_data.size(), 0);
  boost::mpi::communicator communicator;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (communicator.rank() == 0) {
    taskDataPar->inputs_count.push_back(static_cast<int>(input_data.size()));
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.push_back(static_cast<int>(output_data.size()));
  }

  chastov_v_bubble_sort::TestMPITaskParallel<int> parallel_task(taskDataPar);

  ASSERT_TRUE(parallel_task.validation());
  ASSERT_TRUE(parallel_task.pre_processing());
  ASSERT_TRUE(parallel_task.run());
  ASSERT_TRUE(parallel_task.post_processing());

  if (communicator.rank() == 0) {
    EXPECT_EQ((std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}), output_data);
  }
}

TEST(chastov_v_bubble_sort, test_int_rand_120) {
  const size_t massLen = 120;
  std::srand(std::time(nullptr));

  std::vector<int> inputData(massLen);
  for (size_t i = 0; i < massLen; ++i) {
    inputData[i] = std::rand() * static_cast<int>(std::pow(-1, std::rand()));
  }

  std::vector<int> outputData(massLen);

  boost::mpi::communicator mpiWorld;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (mpiWorld.rank() == 0) {
    taskDataPar->inputs_count.push_back(inputData.size());
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(inputData.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(outputData.data()));
    taskDataPar->outputs_count.push_back(outputData.size());
  }

  chastov_v_bubble_sort::TestMPITaskParallel<int> mpiTask(taskDataPar);

  ASSERT_TRUE(mpiTask.validation());

  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();

  if (mpiWorld.rank() == 0) {
    std::sort(inputData.begin(), inputData.end());
    int mismatchedCount = 0;
    for (size_t i = 0; i < massLen; ++i) {
      if (outputData[i] != inputData[i]) {
        ++mismatchedCount;
      }
    }

    ASSERT_EQ(mismatchedCount, 0);
  }
}

TEST(chastov_v_bubble_sort, test_int_rand_1200) {
  const size_t massLen = 1200;
  std::srand(std::time(nullptr));

  std::vector<int> inputData(massLen);
  for (size_t i = 0; i < massLen; ++i) {
    inputData[i] = std::rand() * static_cast<int>(std::pow(-1, std::rand()));
  }

  std::vector<int> outputData(massLen);

  boost::mpi::communicator mpiWorld;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (mpiWorld.rank() == 0) {
    taskDataPar->inputs_count.push_back(inputData.size());
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(inputData.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(outputData.data()));
    taskDataPar->outputs_count.push_back(outputData.size());
  }

  chastov_v_bubble_sort::TestMPITaskParallel<int> mpiTask(taskDataPar);

  ASSERT_TRUE(mpiTask.validation());

  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();

  if (mpiWorld.rank() == 0) {
    std::sort(inputData.begin(), inputData.end());
    int mismatchedCount = 0;
    for (size_t i = 0; i < massLen; ++i) {
      if (outputData[i] != inputData[i]) {
        ++mismatchedCount;
      }
    }

    ASSERT_EQ(mismatchedCount, 0);
  }
}

TEST(chastov_v_bubble_sort, test_double_rand_120) {
  const size_t massLen = 120;
  std::srand(std::time(nullptr));

  std::vector<double> inputData(massLen);
  for (size_t i = 0; i < massLen; ++i) {
    inputData[i] = static_cast<double>(std::rand()) * std::pow(-1, std::rand());
  }

  std::vector<double> outputData(massLen);

  boost::mpi::communicator mpiWorld;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (mpiWorld.rank() == 0) {
    taskDataPar->inputs_count.push_back(inputData.size());
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(inputData.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(outputData.data()));
    taskDataPar->outputs_count.push_back(outputData.size());
  }

  chastov_v_bubble_sort::TestMPITaskParallel<double> mpiTask(taskDataPar);

  ASSERT_TRUE(mpiTask.validation());

  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();

  if (mpiWorld.rank() == 0) {
    std::sort(inputData.begin(), inputData.end());

    int mismatchedCount = 0;
    for (size_t i = 0; i < massLen; ++i) {
      if (outputData[i] != inputData[i]) {
        ++mismatchedCount;
      }
    }

    ASSERT_EQ(mismatchedCount, 0);
  }
}

TEST(chastov_v_bubble_sort, test_double_rand_1200) {
  const size_t massLen = 1200;
  std::srand(std::time(nullptr));

  std::vector<double> inputData(massLen);
  for (size_t i = 0; i < massLen; ++i) {
    inputData[i] = static_cast<double>(std::rand()) * std::pow(-1, std::rand());
  }

  std::vector<double> outputData(massLen);

  boost::mpi::communicator mpiWorld;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (mpiWorld.rank() == 0) {
    taskDataPar->inputs_count.push_back(inputData.size());
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(inputData.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(outputData.data()));
    taskDataPar->outputs_count.push_back(outputData.size());
  }

  chastov_v_bubble_sort::TestMPITaskParallel<double> mpiTask(taskDataPar);

  ASSERT_TRUE(mpiTask.validation());

  mpiTask.pre_processing();
  mpiTask.run();
  mpiTask.post_processing();

  if (mpiWorld.rank() == 0) {
    std::sort(inputData.begin(), inputData.end());

    int mismatchedCount = 0;
    for (size_t i = 0; i < massLen; ++i) {
      if (outputData[i] != inputData[i]) {
        ++mismatchedCount;
      }
    }

    ASSERT_EQ(mismatchedCount, 0);
  }
}

TEST(chastov_v_bubble_sort, test_mass_identical_values) {
  const size_t massLen = 100;
  std::srand(static_cast<unsigned int>(std::time(nullptr)));
  const int constantValue = std::rand();
  std::vector<int> inputData(massLen, constantValue);
  std::vector<int> sortedData(massLen);

  boost::mpi::communicator mpiWorld;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (mpiWorld.rank() == 0) {
    taskDataPar->inputs_count.push_back(inputData.size());
    taskDataPar->inputs.push_back(reinterpret_cast<uint8_t *>(inputData.data()));
    taskDataPar->outputs.push_back(reinterpret_cast<uint8_t *>(sortedData.data()));
    taskDataPar->outputs_count.push_back(sortedData.size());
  }

  chastov_v_bubble_sort::TestMPITaskParallel<int> parallelTask(taskDataPar);

  ASSERT_TRUE(parallelTask.validation());

  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (mpiWorld.rank() == 0) {
    int mismatches = 0;
    for (size_t i = 0; i < massLen; ++i) {
      if (sortedData[i] != inputData[i]) {
        ++mismatches;
      }
    }

    ASSERT_EQ(mismatches, 0);
  }
}