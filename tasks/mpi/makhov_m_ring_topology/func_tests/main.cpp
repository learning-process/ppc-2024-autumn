// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/makhov_m_ring_topology/include/ops_mpi.hpp"
//TEST(makhov_m_ring_topology, RandVectorZeroSize) {
//  boost::mpi::communicator world;
//  std::random_device dev;
//  std::mt19937 gen(dev());
//  size_t size = 0;
//  std::vector<int32_t> input_vector(size);
//  std::vector<int32_t> output_vector(size);
//  std::vector<int32_t> sequence(world.size() + 1);
//  std::vector<int32_t> reference_sequence(world.size() + 1);
//  int32_t min = 0;
//  int32_t max = 9;
//
//  for (size_t i = 0; i < size; i++) {
//    input_vector[i] = (int32_t)(min + gen() % (max - min + 1));
//  }
//
//  for (size_t i = 0; i < (size_t)world.size(); i++) {
//    reference_sequence[i] = i;
//  }
//
//  // Create TaskData
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//
//  if (world.rank() == 0) {
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
//    taskDataPar->inputs_count.emplace_back(input_vector.size());
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
//    taskDataPar->outputs_count.emplace_back(size);
//    taskDataPar->outputs_count.emplace_back(world.size() + 1);
//  }
//
//  // Create Task
//  makhov_m_ring_topology::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
//  ASSERT_TRUE(testMpiTaskParallel.validation());
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//
//  if (world.rank() == 0) {
//    ASSERT_EQ(input_vector, output_vector);
//    ASSERT_EQ(sequence, reference_sequence);
//  }
//}

TEST(makhov_m_ring_topology, RandVectorSize1) {
  boost::mpi::communicator world;
  std::random_device dev;
  std::mt19937 gen(dev());
  size_t size = 1;
  std::vector<int32_t> input_vector(size);
  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);
  std::vector<int32_t> reference_sequence(world.size() + 1);
  int32_t min = 0;
  int32_t max = 9;

  for (size_t i = 0; i < size; i++) {
    input_vector[i] = (int32_t)(min + gen() % (max - min + 1));
  }

  for (size_t i = 0; i < (size_t)world.size(); i++) {
    reference_sequence[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.size() + 1);
  }

  // Create Task
  makhov_m_ring_topology::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(input_vector, output_vector);
    ASSERT_EQ(sequence, reference_sequence);
  }
}

TEST(makhov_m_ring_topology, RandVectorSize10) {
  boost::mpi::communicator world;
  std::random_device dev;
  std::mt19937 gen(dev());
  size_t size = 10;
  std::vector<int32_t> input_vector(size);
  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);
  std::vector<int32_t> reference_sequence(world.size() + 1);
  int32_t min = 0;
  int32_t max = 9;

  for (size_t i = 0; i < size; i++) {
    input_vector[i] = (int32_t)(min + gen() % (max - min + 1));
  }

  for (size_t i = 0; i < (size_t)world.size(); i++) {
    reference_sequence[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.size() + 1);
  }

  // Create Task
  makhov_m_ring_topology::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(input_vector, output_vector);
    ASSERT_EQ(sequence, reference_sequence);
  }
}

TEST(makhov_m_ring_topology, RandVectorSize1000) {
  boost::mpi::communicator world;
  std::random_device dev;
  std::mt19937 gen(dev());
  size_t size = 1000;
  std::vector<int32_t> input_vector(size);
  std::vector<int32_t> output_vector(size);
  std::vector<int32_t> sequence(world.size() + 1);
  std::vector<int32_t> reference_sequence(world.size() + 1);
  int32_t min = 0;
  int32_t max = 9;

  for (size_t i = 0; i < size; i++) {
    input_vector[i] = (int32_t)(min + gen() % (max - min + 1));
  }

  for (size_t i = 0; i < (size_t)world.size(); i++) {
    reference_sequence[i] = i;
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    taskDataPar->inputs_count.emplace_back(input_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(output_vector.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(sequence.data()));
    taskDataPar->outputs_count.emplace_back(size);
    taskDataPar->outputs_count.emplace_back(world.size() + 1);
  }

  // Create Task
  makhov_m_ring_topology::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(input_vector, output_vector);
    ASSERT_EQ(sequence, reference_sequence);
  }
}
