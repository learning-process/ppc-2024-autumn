#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <random>
#include <vector>

#include "mpi/somov_i_bitwise_sorting_batcher_merge/include/ops_mpi.hpp"

namespace somov_i_bitwise_sorting_batcher_merge_mpi {

std::vector<double> create_random_vector(int size, double mean = 3.0, double stddev = 300.0) {
  std::normal_distribution<double> norm_dist(mean, stddev);

  std::random_device rand_dev;
  std::mt19937 rand_engine(rand_dev());

  std::vector<double> tmp(size);
  for (int i = 0; i < size; i++) {
    tmp[i] = norm_dist(rand_engine);
  }
  return tmp;
}

}  // namespace somov_i_bitwise_sorting_batcher_merge_mpi

TEST(somov_i_bitwise_sorting_batcher_merge_MPI, test_sorting_parallel_and_sequential_20_elements) {
  boost::mpi::communicator world;
  std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi::create_random_vector(20);
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> out1(in.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 1; i < out.size(); i++) {
      ASSERT_NEAR(out[i], out1[i], 1e-10);
      ASSERT_TRUE(out[i - 1] <= out[i]);
    }
  }
}

TEST(somov_i_bitwise_sorting_batcher_merge_MPI, test_sorting_parallel_and_sequential_20_elements_reverse) {
  boost::mpi::communicator world;
  std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi::create_random_vector(20);
  std::reverse(in.begin(), in.end());
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> out1(in.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 1; i < out.size(); i++) {
      ASSERT_NEAR(out[i], out1[i], 1e-10);
      ASSERT_TRUE(out[i - 1] <= out[i]);
    }
  }
}

TEST(somov_i_bitwise_sorting_batcher_merge_MPI, test_sorting_parallel_and_sequential_10001_elements_reverse) {
  boost::mpi::communicator world;
  std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi::create_random_vector(10001);
  std::reverse(in.begin(), in.end());
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> out1(in.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 1; i < out.size(); i++) {
      ASSERT_NEAR(out[i], out1[i], 1e-10);
      ASSERT_TRUE(out[i - 1] <= out[i]);
    }
  }
}

TEST(somov_i_bitwise_sorting_batcher_merge_MPI, test_sorting_parallel_and_sequential_1001_elements) {
  boost::mpi::communicator world;
  std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi::create_random_vector(1001);
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> out1(in.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 1; i < out.size(); i++) {
      ASSERT_NEAR(out[i], out1[i], 1e-10);
      ASSERT_TRUE(out[i - 1] <= out[i]);
    }
  }
}

TEST(somov_i_bitwise_sorting_batcher_merge_MPI, test_sorting_parallel_and_sequential_1_element) {
  boost::mpi::communicator world;
  std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi::create_random_vector(1);
  std::vector<double> out(in.size(), 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<double> out1(in.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskDataSeq->outputs_count.emplace_back(out1.size());

    somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    for (size_t i = 1; i < out.size(); i++) {
      ASSERT_NEAR(out[i], out1[i], 1e-10);
      ASSERT_TRUE(out[i - 1] <= out[i]);
    }
  }
}

TEST(somov_i_bitwise_sorting_batcher_merge_MPI, test_sorting_parallel_and_sequential_power_of_two) {
  boost::mpi::communicator world;
  std::vector<int> sizes = {2, 4, 8, 16, 32, 64};

  for (int size : sizes) {
    std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi::create_random_vector(size);
    std::vector<double> out(in.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      taskDataPar->inputs_count.emplace_back(in.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }

    somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      std::vector<double> out1(in.size(), 0);

      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      taskDataSeq->inputs_count.emplace_back(in.size());
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
      taskDataSeq->outputs_count.emplace_back(out1.size());

      somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
      ASSERT_EQ(testMpiTaskSequential.validation(), true);
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();
      for (size_t i = 1; i < out.size(); i++) {
        ASSERT_NEAR(out[i], out1[i], 1e-10);
        ASSERT_TRUE(out[i - 1] <= out[i]);
      }
    }
  }
}

TEST(somov_i_bitwise_sorting_batcher_merge_MPI, test_sorting_parallel_and_sequential_power_of_three) {
  boost::mpi::communicator world;
  std::vector<int> sizes = {3, 9, 27, 81, 243};

  for (int size : sizes) {
    std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi::create_random_vector(size);
    std::vector<double> out(in.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      taskDataPar->inputs_count.emplace_back(in.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }

    somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      std::vector<double> out1(in.size(), 0);

      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      taskDataSeq->inputs_count.emplace_back(in.size());
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
      taskDataSeq->outputs_count.emplace_back(out1.size());

      somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
      ASSERT_EQ(testMpiTaskSequential.validation(), true);
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();
      for (size_t i = 1; i < out.size(); i++) {
        ASSERT_NEAR(out[i], out1[i], 1e-10);
        ASSERT_TRUE(out[i - 1] <= out[i]);
      }
    }
  }
}

TEST(somov_i_bitwise_sorting_batcher_merge_MPI, test_sorting_parallel_and_sequential_prime_sizes) {
  boost::mpi::communicator world;
  std::vector<int> sizes = {5, 7, 11, 13, 17, 19};

  for (int size : sizes) {
    std::vector<double> in = somov_i_bitwise_sorting_batcher_merge_mpi::create_random_vector(size);
    std::vector<double> out(in.size(), 0);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      taskDataPar->inputs_count.emplace_back(in.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
      taskDataPar->outputs_count.emplace_back(out.size());
    }

    somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), true);
    testMpiTaskParallel.pre_processing();
    testMpiTaskParallel.run();
    testMpiTaskParallel.post_processing();

    if (world.rank() == 0) {
      std::vector<double> out1(in.size(), 0);

      std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
      taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
      taskDataSeq->inputs_count.emplace_back(in.size());
      taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
      taskDataSeq->outputs_count.emplace_back(out1.size());

      somov_i_bitwise_sorting_batcher_merge_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
      ASSERT_EQ(testMpiTaskSequential.validation(), true);
      testMpiTaskSequential.pre_processing();
      testMpiTaskSequential.run();
      testMpiTaskSequential.post_processing();
      for (size_t i = 1; i < out.size(); i++) {
        ASSERT_NEAR(out[i], out1[i], 1e-6);
        ASSERT_TRUE(out[i - 1] <= out[i]);
      }
    }
  }
}
