// Filateva Elizaveta Radix Sort
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/filateva_e_radix_sort/include/ops_mpi.hpp"

namespace filateva_e_radix_sort_mpi {

void GeneratorVector(std::vector<int> &vec, int max_z, int min_z) {
  std::random_device dev;
  std::mt19937 gen(dev());
  for (unsigned long i = 0; i < vec.size(); i++) {
    vec[i] = gen() % (max_z - min_z + 1) + min_z;
  }
}

}  // namespace filateva_e_radix_sort_mpi

TEST(filateva_e_radix_sort_mpi, test_size_3) {
  boost::mpi::communicator world;
  int size = 3;
  int max_z = 10;
  int min_z = -10;
  std::vector<int> vec;
  std::vector<int> answer;
  std::vector<int> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    filateva_e_radix_sort_mpi::GeneratorVector(vec, max_z, min_z);
    tResh = vec;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_radix_sort_mpi::RadixSort radixSort(taskData);

  ASSERT_TRUE(radixSort.validation());
  radixSort.pre_processing();
  radixSort.run();
  radixSort.post_processing();

  if (world.rank() == 0) {
    std::sort(tResh.begin(), tResh.end());

    EXPECT_EQ(answer.size(), tResh.size());
    for (int i = 0; i < size; i++) {
      EXPECT_EQ(answer[i], tResh[i]);
    }
  }
}

TEST(filateva_e_radix_sort_mpi, test_size_10) {
  boost::mpi::communicator world;
  int size = 10;
  int max_z = 10;
  int min_z = -10;
  std::vector<int> vec;
  std::vector<int> answer;
  std::vector<int> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    filateva_e_radix_sort_mpi::GeneratorVector(vec, max_z, min_z);
    tResh = vec;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_radix_sort_mpi::RadixSort radixSort(taskData);

  ASSERT_TRUE(radixSort.validation());
  radixSort.pre_processing();
  radixSort.run();
  radixSort.post_processing();

  if (world.rank() == 0) {
    std::sort(tResh.begin(), tResh.end());

    EXPECT_EQ(answer.size(), tResh.size());
    for (int i = 0; i < size; i++) {
      EXPECT_EQ(answer[i], tResh[i]);
    }
  }
}

TEST(filateva_e_radix_sort_mpi, test_size_30) {
  boost::mpi::communicator world;
  int size = 30;
  int max_z = 100;
  int min_z = -100;
  std::vector<int> vec;
  std::vector<int> answer;
  std::vector<int> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    filateva_e_radix_sort_mpi::GeneratorVector(vec, max_z, min_z);
    tResh = vec;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_radix_sort_mpi::RadixSort radixSort(taskData);

  ASSERT_TRUE(radixSort.validation());
  radixSort.pre_processing();
  radixSort.run();
  radixSort.post_processing();

  if (world.rank() == 0) {
    std::sort(tResh.begin(), tResh.end());

    EXPECT_EQ(answer.size(), tResh.size());
    for (int i = 0; i < size; i++) {
      EXPECT_EQ(answer[i], tResh[i]);
    }
  }
}

TEST(filateva_e_radix_sort_mpi, test_size_100) {
  boost::mpi::communicator world;
  int size = 100;
  int max_z = 1000;
  int min_z = -1000;
  std::vector<int> vec;
  std::vector<int> answer;
  std::vector<int> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    filateva_e_radix_sort_mpi::GeneratorVector(vec, max_z, min_z);
    tResh = vec;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_radix_sort_mpi::RadixSort radixSort(taskData);

  ASSERT_TRUE(radixSort.validation());
  radixSort.pre_processing();
  radixSort.run();
  radixSort.post_processing();

  if (world.rank() == 0) {
    std::sort(tResh.begin(), tResh.end());

    EXPECT_EQ(answer.size(), tResh.size());
    for (int i = 0; i < size; i++) {
      EXPECT_EQ(answer[i], tResh[i]);
    }
  }
}

TEST(filateva_e_radix_sort_mpi, test_size_211) {
  boost::mpi::communicator world;
  int size = 211;
  int max_z = 100000;
  int min_z = -100000;
  std::vector<int> vec;
  std::vector<int> answer;
  std::vector<int> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    filateva_e_radix_sort_mpi::GeneratorVector(vec, max_z, min_z);
    tResh = vec;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_radix_sort_mpi::RadixSort radixSort(taskData);

  ASSERT_TRUE(radixSort.validation());
  radixSort.pre_processing();
  radixSort.run();
  radixSort.post_processing();

  if (world.rank() == 0) {
    std::sort(tResh.begin(), tResh.end());

    EXPECT_EQ(answer.size(), tResh.size());
    for (int i = 0; i < size; i++) {
      EXPECT_EQ(answer[i], tResh[i]);
    }
  }
}

TEST(filateva_e_radix_sort_mpi, test_size_different) {
  boost::mpi::communicator world;
  int size = 10;
  std::vector<int> vec;
  std::vector<int> answer;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size + 1);
  }

  filateva_e_radix_sort_mpi::RadixSort radixSort(taskData);

  if (world.rank() == 0) {
    EXPECT_FALSE(radixSort.validation());
  }
}

TEST(filateva_e_radix_sort_mpi, test_size_0) {
  boost::mpi::communicator world;
  int size = 0;
  std::vector<int> vec;
  std::vector<int> answer;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_radix_sort_mpi::RadixSort radixSort(taskData);

  if (world.rank() == 0) {
    EXPECT_FALSE(radixSort.validation());
  }
}

TEST(filateva_e_radix_sort_mpi, less_0) {
  boost::mpi::communicator world;
  int size = 0;
  std::vector<int> vec;
  std::vector<int> answer;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(-size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_radix_sort_mpi::RadixSort radixSort(taskData);

  if (world.rank() == 0) {
    EXPECT_FALSE(radixSort.validation());
  }
}

TEST(filateva_e_radix_sort_mpi, test_revers) {
  boost::mpi::communicator world;
  int size = 100;
  std::vector<int> vec;
  std::vector<int> answer;
  std::vector<int> tResh;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    vec.resize(size);
    answer.resize(size);

    for (int i = 0; i < size; i++) {
      vec[i] = 100 - i;
    }
    tResh = vec;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(vec.data()));
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(answer.data()));
    taskData->inputs_count.emplace_back(size);
    taskData->outputs_count.emplace_back(size);
  }

  filateva_e_radix_sort_mpi::RadixSort radixSort(taskData);

  ASSERT_TRUE(radixSort.validation());
  radixSort.pre_processing();
  radixSort.run();
  radixSort.post_processing();

  if (world.rank() == 0) {
    std::sort(tResh.begin(), tResh.end());

    EXPECT_EQ(answer.size(), tResh.size());
    for (int i = 0; i < size; i++) {
      EXPECT_EQ(answer[i], tResh[i]);
    }
  }
}
