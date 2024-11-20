// Copyright 2023 Nesterov Alexander
// mpiexec -n 4 bin/mpi_func_tests.exe --gtest_filter=kurakin_m_producer_consumer_mpi.*
// cmake --build . --target mpi_func_tests --config RELEASE
// cmake -D USE_SEQ=ON -D USE_MPI=ON -D USE_OMP=ON -D USE_TBB=ON -D USE_STL=ON -D USE_FUNC_TESTS=ON -D USE_PERF_TESTS=ON
// -D CMAKE_BUILD_TYPE=Release ..

#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>
#include <numeric>

#include "mpi/kurakin_m_producer_consumer/include/kurakin_producer_consumer_ops_mpi.hpp"

TEST(kurakin_m_producer_consumer_mpi, Test_prod_half_of_the_proc_data_counts_10) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = world.size() / 2;
    int buffer_size = producer_count;
    data_counts = std::vector<int>(producer_count, 10);
    
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_prod_half_of_the_proc_data_counts_0) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = world.size() / 2;
    int buffer_size = producer_count;
    data_counts = std::vector<int>(producer_count, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_prod_half_of_the_proc_data_counts_rand) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = world.size() / 2;
    int buffer_size = producer_count;
    data_counts = kurakin_m_producer_consumer_mpi::getRandomVector(producer_count, 0, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_prod_less_conc_data_counts_10) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = std::max(1, world.size() / 4);
    int buffer_size = producer_count;
    data_counts = std::vector<int>(producer_count, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_prod_less_conc_data_counts_0) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = std::max(1, world.size() / 4);
    int buffer_size = producer_count;
    data_counts = std::vector<int>(producer_count, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_prod_less_conc_data_counts_rand) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = std::max(1, world.size() / 4);
    int buffer_size = producer_count;
    data_counts = kurakin_m_producer_consumer_mpi::getRandomVector(producer_count, 0, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_prod_more_conc_data_counts_10) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = std::min(world.size() - 1, 3 * world.size() / 4);
    int buffer_size = producer_count;
    data_counts = std::vector<int>(producer_count, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_prod_more_conc_data_counts_0) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = std::min(world.size() - 1, 3 * world.size() / 4);
    int buffer_size = producer_count;
    data_counts = std::vector<int>(producer_count, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_prod_more_conc_data_counts_rand) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = std::min(world.size() - 1, 3 * world.size() / 4);
    int buffer_size = producer_count;
    data_counts = kurakin_m_producer_consumer_mpi::getRandomVector(producer_count, 0, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_buffer_less_count_prod) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = world.size() / 2;
    int buffer_size = std::max(1, producer_count / 2);
    data_counts = kurakin_m_producer_consumer_mpi::getRandomVector(producer_count, 0, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_buffer_more_count_prod) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = world.size() / 2;
    int buffer_size = std::min(world.size() - 1, 3 * producer_count / 2);
    data_counts = kurakin_m_producer_consumer_mpi::getRandomVector(producer_count, 0, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());
  }

  kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(std::accumulate(data_counts.begin(), data_counts.end(), 0), data_counts_sum[0]);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_validation_not_enough_taskData) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = world.size() / 2;
    data_counts = kurakin_m_producer_consumer_mpi::getRandomVector(producer_count, 0, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());

    kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_validation_buffer_size_0) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = world.size() / 2;
    int buffer_size = 0;
    data_counts = kurakin_m_producer_consumer_mpi::getRandomVector(producer_count, 0, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());

    kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_validation_prod_count_0) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = 0;
    int buffer_size = world.size() / 2;
    data_counts = kurakin_m_producer_consumer_mpi::getRandomVector(world.size() / 2, 0, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());

    kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}

TEST(kurakin_m_producer_consumer_mpi, Test_validation_prod_count_world_size) {
  boost::mpi::communicator world;
  std::vector<int> data_counts;
  std::vector<int32_t> data_counts_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int producer_count = world.size();
    int buffer_size = world.size() / 2;
    data_counts = kurakin_m_producer_consumer_mpi::getRandomVector(world.size() / 2, 0, 10);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts.data()));
    taskDataPar->inputs_count.emplace_back(data_counts.size());
    taskDataPar->inputs_count.emplace_back(producer_count);
    taskDataPar->inputs_count.emplace_back(buffer_size);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(data_counts_sum.data()));
    taskDataPar->outputs_count.emplace_back(data_counts_sum.size());

    kurakin_m_producer_consumer_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
