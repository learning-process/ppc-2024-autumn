// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/kondratev_ya_radix_sort_batcher_merge/include/ops_mpi.hpp"
namespace kondratev_ya_radix_sort_batcher_merge_mpi {
std::vector<double> getRandomVector(uint32_t size) {
  std::srand(std::time(nullptr));
  std::vector<double> vec(size);

  double lower_bound = -10000;
  double upper_bound = 10000;
  for (uint32_t i = 0; i < size; i++) {
    vec[i] = lower_bound + std::rand() / (double)RAND_MAX * (upper_bound - lower_bound);
  }
  return vec;
}
}  // namespace kondratev_ya_radix_sort_batcher_merge_mpi

TEST(kondratev_ya_radix_sort_batcher_merge_mpi, basic) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> out;
  std::vector<double> out1;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    uint32_t size = 12;
    in.assign({8, 2, 5, 10, 1, 7, 3, 12, 6, 11, 4, 9});
    out.resize(size);
    out1.resize(size);

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel testTaskParallel(taskDataMPI);
  testTaskParallel.validation();
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskData->outputs_count.emplace_back(out1.size());

    kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskSequential testTaskSequential(taskData);
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], out1[0]);
    for (uint32_t i = 1; i < out.size(); i++) {
      ASSERT_LE(out[i - 1], out[i]);
      ASSERT_EQ(out[i], out1[i]);
    }
  }
}

TEST(kondratev_ya_radix_sort_batcher_merge_mpi, empty) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> out;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel testTaskParallel(taskDataMPI);
  testTaskParallel.validation();
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());

    kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskSequential testTaskSequential(taskData);
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();
  }
}

TEST(kondratev_ya_radix_sort_batcher_merge_mpi, scalar) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> out;
  std::vector<double> out1;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    uint32_t size = 1;
    in = kondratev_ya_radix_sort_batcher_merge_mpi::getRandomVector(size);
    out.resize(size);
    out1.resize(size);

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel testTaskParallel(taskDataMPI);
  testTaskParallel.validation();
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskData->outputs_count.emplace_back(out1.size());

    kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskSequential testTaskSequential(taskData);
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], out1[0]);
    for (uint32_t i = 1; i < out.size(); i++) {
      ASSERT_LE(out[i - 1], out[i]);
      ASSERT_EQ(out[i], out1[i]);
    }
  }
}

TEST(kondratev_ya_radix_sort_batcher_merge_mpi, prime) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> out;
  std::vector<double> out1;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    uint32_t size = 239;
    in = kondratev_ya_radix_sort_batcher_merge_mpi::getRandomVector(size);
    out.resize(size);
    out1.resize(size);

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel testTaskParallel(taskDataMPI);
  testTaskParallel.validation();
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskData->outputs_count.emplace_back(out1.size());

    kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskSequential testTaskSequential(taskData);
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], out1[0]);
    for (uint32_t i = 1; i < out.size(); i++) {
      ASSERT_LE(out[i - 1], out[i]);
      ASSERT_EQ(out[i], out1[i]);
    }
  }
}

TEST(kondratev_ya_radix_sort_batcher_merge_mpi, power2) {
  boost::mpi::communicator world;
  std::vector<double> in;
  std::vector<double> out;
  std::vector<double> out1;

  std::shared_ptr<ppc::core::TaskData> taskDataMPI = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    uint32_t size = 256;
    in = kondratev_ya_radix_sort_batcher_merge_mpi::getRandomVector(size);
    out.resize(size);
    out1.resize(size);

    taskDataMPI->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskDataMPI->inputs_count.emplace_back(in.size());
    taskDataMPI->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataMPI->outputs_count.emplace_back(out.size());
  }

  kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskParallel testTaskParallel(taskDataMPI);
  testTaskParallel.validation();
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out1.data()));
    taskData->outputs_count.emplace_back(out1.size());

    kondratev_ya_radix_sort_batcher_merge_mpi::TestMPITaskSequential testTaskSequential(taskData);
    testTaskSequential.validation();
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    ASSERT_EQ(out[0], out1[0]);
    for (uint32_t i = 1; i < out.size(); i++) {
      ASSERT_LE(out[i - 1], out[i]);
      ASSERT_EQ(out[i], out1[i]);
    }
  }
}
