#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "mpi/naumov_b_bubble_sort/include/ops_mpi.hpp"

namespace naumov_b_bubble_sort_mpi {
std::vector<int> getRandomVec(int size) {
  std::vector<int> vec(size);
  for (int &element : vec) {
    element = rand() % 201 - 100;
  }
  return vec;
}
}  // namespace naumov_b_bubble_sort_mpi

TEST(naumov_b_bubble_sort_mpi, Test_SmallArray) {
  boost::mpi::communicator world;

  std::vector<int> input_data = {5, 3, 8, 6, 2};
  std::vector<int> sorted_data = {2, 3, 5, 6, 8};

  std::vector<int> output_data(input_data.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  naumov_b_bubble_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data, sorted_data);
  }
}

TEST(naumov_b_bubble_sort_mpi, Test_LargeArray) {
  boost::mpi::communicator world;

  std::vector<int> input_data = naumov_b_bubble_sort_mpi::getRandomVec(1000);

  std::vector<int> sorted_data = input_data;
  std::sort(sorted_data.begin(), sorted_data.end());

  std::vector<int> output_data(input_data.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  naumov_b_bubble_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data, sorted_data);
  }
}

TEST(naumov_b_bubble_sort_mpi, Test_SingleElement) {
  boost::mpi::communicator world;

  std::vector<int> input_data = {42};
  std::vector<int> sorted_data = {42};

  std::vector<int> output_data(input_data.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  naumov_b_bubble_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data, sorted_data);
  }
}

TEST(naumov_b_bubble_sort_mpi, Test_RepeatedElements) {
  boost::mpi::communicator world;

  std::vector<int> input_data = {7, 3, 7, 1, 3, 1, 7};
  std::vector<int> sorted_data = {1, 1, 3, 3, 7, 7, 7};

  std::vector<int> output_data(input_data.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  naumov_b_bubble_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data, sorted_data);
  }
}

TEST(naumov_b_bubble_sort_mpi, Test_NegativeNumbers) {
  boost::mpi::communicator world;

  std::vector<int> input_data = {-5, -1, -3, -4, -2};
  std::vector<int> sorted_data = {-5, -4, -3, -2, -1};

  std::vector<int> output_data(input_data.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  naumov_b_bubble_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data, sorted_data);
  }
}

TEST(naumov_b_bubble_sort_mpi, Test_AlternatingHighLow) {
  boost::mpi::communicator world;

  std::vector<int> input_data = {100, -100, 50, -50, 0};
  std::vector<int> sorted_data = {-100, -50, 0, 50, 100};

  std::vector<int> output_data(input_data.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  naumov_b_bubble_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data, sorted_data);
  }
}

TEST(naumov_b_bubble_sort_mpi, Test_IntLimits) {
  boost::mpi::communicator world;

  std::vector<int> input_data = {INT_MAX, 0, INT_MIN, -1, 1};
  std::vector<int> sorted_data = {INT_MIN, -1, 0, 1, INT_MAX};

  std::vector<int> output_data(input_data.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  naumov_b_bubble_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data, sorted_data);
  }
}

TEST(naumov_b_bubble_sort_mpi, Test_ReverseOrder) {
  boost::mpi::communicator world;

  std::vector<int> input_data = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> sorted_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

  std::vector<int> output_data(input_data.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_data.data()));
    taskDataPar->outputs_count.emplace_back(output_data.size());
  }

  naumov_b_bubble_sort_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(output_data, sorted_data);
  }
}

TEST(naumov_b_bubble_sort_mpi, Test_empty_array) {
  const size_t length = 0;
  std::vector<int> in(length);
  std::vector<int> out(length);

  boost::mpi::communicator world;
  std::shared_ptr<ppc::core::TaskData> tmpPar = std::make_shared<ppc::core::TaskData>();

  tmpPar->inputs_count.emplace_back(in.size());
  tmpPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  tmpPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  tmpPar->outputs_count.emplace_back(out.size());

  naumov_b_bubble_sort_mpi::TestMPITaskParallel tmpTaskPar(tmpPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(tmpTaskPar.validation());
  }
}
