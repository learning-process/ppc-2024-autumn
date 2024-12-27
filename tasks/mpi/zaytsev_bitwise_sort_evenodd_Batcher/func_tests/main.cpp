// Copyright 2024 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <random>
#include <vector>

#include "mpi/zaytsev_bitwise_sort_evenodd_Batcher/include/ops_mpi.hpp"

static std::vector<int> getRandomVector(int sz, int minVal = -50, int maxVal = 50) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (maxVal - minVal + 1) + minVal;
  }
  return vec;
}

static std::vector<int> generatePowers(int sz, int base) {
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = std::pow(base, i);
  }
  return vec;
}

static std::vector<int> generatePrimeNumbers(std::size_t sz) {
  std::vector<int> primes;
  int num = 2;
  while (primes.size() < sz) {
    bool is_prime = true;
    for (int i = 2; i * i <= num; i++) {
      if (num % i == 0) {
        is_prime = false;
        break;
      }
    }
    if (is_prime) {
      primes.push_back(num);
    }
    num++;
  }
  return primes;
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher, CorrectSorting) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {4, -2, 7, -5, 3, 8, -1, 0, 6, -9, -10, 13};
  std::vector<int> global_result(test_vector.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequential_result(test_vector.size(), 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_result.data()));
    taskDataSeq->outputs_count.emplace_back(sequential_result.size());

    zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result, sequential_result);
  }
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher, AlreadySorted) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {-9, -5, -2, -1, 0, 3, 4, 6, 7, 8, 9, 10};
  std::vector<int> global_result(test_vector.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequential_result(test_vector.size(), 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_result.data()));
    taskDataSeq->outputs_count.emplace_back(sequential_result.size());

    zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result, sequential_result);
  }
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher, EmptyVector) {
  boost::mpi::communicator world;
  std::vector<int> test_vector;
  std::vector<int> global_result(test_vector.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(global_result.empty());
  }
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher, RandomVector) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = getRandomVector(120);
  std::vector<int> global_result(test_vector.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequential_result(test_vector.size(), 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_result.data()));
    taskDataSeq->outputs_count.emplace_back(sequential_result.size());

    zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result, sequential_result);
  }
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher, RevSorted) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {6, 5, 4, 3, 2, 1, -1, -2, -3, -4, -5, -6};
  std::vector<int> global_result(test_vector.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequential_result(test_vector.size(), 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_result.data()));
    taskDataSeq->outputs_count.emplace_back(sequential_result.size());

    zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result, sequential_result);
  }
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher, PowersOfTwo) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = generatePowers(12, 2);
  std::vector<int> global_result(test_vector.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequential_result(test_vector.size(), 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_result.data()));
    taskDataSeq->outputs_count.emplace_back(sequential_result.size());

    zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result, sequential_result);
  }
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher, PowersOfThree) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = generatePowers(12, 3);
  std::vector<int> global_result(test_vector.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequential_result(test_vector.size(), 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_result.data()));
    taskDataSeq->outputs_count.emplace_back(sequential_result.size());

    zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result, sequential_result);
  }
}

TEST(zaytsev_bitwise_sort_evenodd_Batcher, PrimeNumber) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = generatePrimeNumbers(12);
  std::vector<int> global_result(test_vector.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> sequential_result(test_vector.size(), 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataSeq->inputs_count.emplace_back(test_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_result.data()));
    taskDataSeq->outputs_count.emplace_back(sequential_result.size());

    zaytsev_bitwise_sort_evenodd_Batcher::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(global_result, sequential_result);
  }
}