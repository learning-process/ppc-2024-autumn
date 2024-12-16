// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <functional>
#include <random>
#include <vector>

#include "mpi/tsatsyn_a_vector_dot_product/include/ops_mpi.hpp"
std::vector<int> toGetRandomVector(int size) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> v(size);
  for (int i = 0; i < size; i++) {
    v[i] = gen() % 200 + gen() % 10;
  }
  return v;
}

TEST(tsatsyn_a_vector_dot_product_mpi, Test_Random_Scalar) {
  boost::mpi::communicator world;
  std::vector<int> v1 = toGetRandomVector(60);
  std::vector<int> v2 = toGetRandomVector(60);
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
  }
}
TEST(tsatsyn_a_vector_dot_product_mpi, 10xTest_Random_Scalar) {
  boost::mpi::communicator world;
  std::vector<int> v1 = toGetRandomVector(120);
  std::vector<int> v2 = toGetRandomVector(120);
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> res2(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));

    taskDataSeq->inputs_count.emplace_back(v1.size());
    taskDataSeq->inputs_count.emplace_back(v2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res2.data()));
    taskDataSeq->outputs_count.emplace_back(res2.size());
    tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(res2[0], res[0]);
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
  }
}
TEST(tsatsyn_a_vector_dot_product_mpi, 100xTest_Random_Scalar) {
  boost::mpi::communicator world;
  std::vector<int> v1 = toGetRandomVector(1200);
  std::vector<int> v2 = toGetRandomVector(1200);
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
  }
}
TEST(tsatsyn_a_vector_dot_product_mpi, 1000xTest_Random_Scalar) {
  boost::mpi::communicator world;
  std::vector<int> v1 = toGetRandomVector(12000);
  std::vector<int> v2 = toGetRandomVector(12000);
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v2.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);

  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
  }
}