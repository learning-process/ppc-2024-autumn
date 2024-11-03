// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/tsatsyn_a_vector_dot_product/include/ops_mpi.hpp"
std::vector<int> toGetRandomVector(int size) {
  std::vector<int> vector(size);
  std::srand((time(nullptr)));
  for (int i = 0; i < size; ++i) {
    vector[i] = std::rand() % 200 + std::rand() % 10;
  }
  return vector;
}

//TEST(tsatsyn_a_vector_dot_product_mpi, Test_Scalar_Negative_Value) {
//  boost::mpi::communicator world;
//  std::vector<int> v1 = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
//  std::vector<int> v2 = {99, 88, 77, 66, 55, 44, 33, 22, 11, 10, -11, -22};
//  std::vector<int> res(1, 0);
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//  if (world.rank() == 0) {
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
//    taskDataPar->inputs_count.emplace_back(v1.size());
//
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
//    taskDataPar->inputs_count.emplace_back(v2.size());
//
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
//    taskDataPar->outputs_count.emplace_back(res.size());
//  }
//  tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
//  ASSERT_EQ(testMpiTaskParallel.validation(), true);
//  testMpiTaskParallel.pre_processing();
//  testMpiTaskParallel.run();
//  testMpiTaskParallel.post_processing();
//  ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
//}
TEST(tsatsyn_a_vector_dot_product_mpi, Test_Scalar_Positive_Value) {
  boost::mpi::communicator world;
  std::vector<int> v1 = {7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84};
  std::vector<int> v2 = {9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108};
  std::vector<int> res(1, 0);
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
  ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
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
  ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
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
  ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
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
  ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
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
  ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
}
TEST(tsatsyn_a_vector_dot_product_mpi, Test_Negative_Validation) {
  boost::mpi::communicator world;
  std::vector<int> v1 = {1, 2, 3};
  std::vector<int> v2 = {4, 5, 6, 0};
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
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}
//TEST(tsatsyn_a_vector_dot_product_mpi, Test_Positive_Validation) {
//  boost::mpi::communicator world;
//  std::vector<int> v1 = {1, 2, 3};
//  std::vector<int> v2 = {4, 5, 6};
//  std::vector<int32_t> res(1, 0);
//  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//  if (world.rank() == 0) {
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
//    taskDataPar->inputs_count.emplace_back(v1.size());
//
//    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
//    taskDataPar->inputs_count.emplace_back(v2.size());
//
//    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
//    taskDataPar->outputs_count.emplace_back(res.size());
//  }
//  tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
//  ASSERT_EQ(testMpiTaskParallel.validation(), true);
//}