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

// TEST(tsatsyn_a_vector_dot_product_mpi, Test_Scalar_Negative_Value) {
//   boost::mpi::communicator world;
//   std::vector<int> v1 = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
//   std::vector<int> v2 = {99, 88, 77, 66, 55, 44, 33, 22, 11, 10, -11, -22};
//   std::vector<int> res(1, 0);
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//   if (world.rank() == 0) {
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
//     taskDataPar->inputs_count.emplace_back(v1.size());
//
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
//     taskDataPar->inputs_count.emplace_back(v2.size());
//
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
//     taskDataPar->outputs_count.emplace_back(res.size());
//   }
//   tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
//   ASSERT_EQ(testMpiTaskParallel.validation(), true);
//   testMpiTaskParallel.pre_processing();
//   testMpiTaskParallel.run();
//   testMpiTaskParallel.post_processing();
//   ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
// }
TEST(tsatsyn_a_vector_dot_product_mpi, Test_Scalar_Positive_Value) {
  boost::mpi::communicator world;
  std::vector<int> v1 = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                         41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60};
  std::vector<int> v2 = {60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41,
                         40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
                         20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4,  3,  2,  1};
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
  if (world.rank() == 0) {
    ASSERT_EQ(tsatsyn_a_vector_dot_product_mpi::resulting(v1, v2), res[0]);
  }
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
//TEST(tsatsyn_a_vector_dot_product_mpi, Test_Negative_Validation) {
//  boost::mpi::communicator world;
//  std::vector<int> v1 = {1, 2, 3};
//  std::vector<int> v2 = {4, 5, 6, 0};
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
//  ASSERT_EQ(testMpiTaskParallel.validation(), false);
//}
// TEST(tsatsyn_a_vector_dot_product_mpi, Test_Positive_Validation) {
//   boost::mpi::communicator world;
//   std::vector<int> v1 = {1, 2, 3};
//   std::vector<int> v2 = {4, 5, 6};
//   std::vector<int32_t> res(1, 0);
//   std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
//   if (world.rank() == 0) {
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v1.data()));
//     taskDataPar->inputs_count.emplace_back(v1.size());
//
//     taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(v2.data()));
//     taskDataPar->inputs_count.emplace_back(v2.size());
//
//     taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(res.data()));
//     taskDataPar->outputs_count.emplace_back(res.size());
//   }
//   tsatsyn_a_vector_dot_product_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
//   ASSERT_EQ(testMpiTaskParallel.validation(), true);
// }