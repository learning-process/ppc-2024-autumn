// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/tsatsyn_a_vector_dot_product/include/ops_mpi.hpp"
std::vector<int> GetRandomVector(int size) {
  std::vector<int> vector(size);
  std::srand((time(NULL)));
  for (int i = 0; i < size; ++i) {
    vector[i] = std::rand() % 100 + 1;
  }
  return vector;
}
TEST(tsatsyn_a_vector_dot_product_mpi, Test_Random_Scalar) {
  boost::mpi::communicator world;
  std::vector<int> v1 = GetRandomVector(3);
  std::vector<int> v2 = GetRandomVector(3);
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->inputs_count.emplace_back(v2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  if (world.rank() == 0) {
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataSeq->inputs_count.emplace_back(v1.size());
    taskDataSeq->inputs_count.emplace_back(v2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
  }
}
TEST(tsatsyn_a_vector_dot_product_mpi, Test_Sum) {
  boost::mpi::communicator world;
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {4, 5, 6};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
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
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataSeq->inputs_count.emplace_back(v1.size());
    taskDataSeq->inputs_count.emplace_back(v2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
    ASSERT_EQ(reference[0], res[0]);
  }
}

TEST(tsatsyn_a_vector_dot_product_mpi, Test_Negative_Validation) {
  boost::mpi::communicator world;
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {4, 5, 6, 0};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->inputs_count.emplace_back(v2.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
    taskDataPar->outputs_count.emplace_back(res.size());
  }
  tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
  if (world.rank() == 0) {
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataSeq->inputs_count.emplace_back(v1.size());
    taskDataSeq->inputs_count.emplace_back(v2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}

TEST(tsatsyn_a_vector_dot_product_mpi, Test_Scalar_32) {
  boost::mpi::communicator world;
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {4, 5, 6};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
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
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataSeq->inputs_count.emplace_back(v1.size());
    taskDataSeq->inputs_count.emplace_back(v2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
    ASSERT_EQ(reference[0], res[0]);
  }
}

TEST(tsatsyn_a_vector_dot_product_mpi, Test_Scalar_28) {
  boost::mpi::communicator world;
  std::vector<int> v1 = {6, 4, 5};
  std::vector<int> v2 = {1, 3, 2};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
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
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataSeq->inputs_count.emplace_back(v1.size());
    taskDataSeq->inputs_count.emplace_back(v2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
    ASSERT_EQ(reference[0], res[0]);
  }
}

TEST(tsatsyn_a_vector_dot_product_mpi, Test_Scalar_95) {
  boost::mpi::communicator world;
  std::vector<int> v1 = {7, 4, 6};
  std::vector<int> v2 = {3, 5, 9};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
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
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataSeq->inputs_count.emplace_back(v1.size());
    taskDataSeq->inputs_count.emplace_back(v2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
    ASSERT_EQ(reference[0], res[0]);
  }
}

TEST(tsatsyn_a_vector_dot_product_mpi, Test_Scalar_2330) {
  boost::mpi::communicator world;
  std::vector<int> v1 = {20, 54, 23};
  std::vector<int> v2 = {32, 10, 50};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
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
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataSeq->inputs_count.emplace_back(v1.size());
    taskDataSeq->inputs_count.emplace_back(v2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
    ASSERT_EQ(reference[0], res[0]);
  }
}

TEST(tsatsyn_a_vector_dot_product_mpi, Test_Scalar_1956) {
  boost::mpi::communicator world;
  std::vector<int> v1 = {12, 100, 50};
  std::vector<int> v2 = {13, 3, 30};
  std::vector<int32_t> res(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
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
    std::vector<int32_t> reference(1, 0);
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
    taskDataSeq->inputs_count.emplace_back(v1.size());
    taskDataSeq->inputs_count.emplace_back(v2.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference.data()));
    taskDataSeq->outputs_count.emplace_back(reference.size());
    tsatsyn_a_vector_dot_product_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
    ASSERT_EQ(reference[0], res[0]);
  }
}