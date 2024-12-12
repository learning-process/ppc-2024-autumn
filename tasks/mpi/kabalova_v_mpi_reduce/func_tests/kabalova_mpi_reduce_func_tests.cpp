// Copyright 2024 Kabalova Valeria
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>

#include "mpi/kabalova_v_mpi_reduce/include/kabalova_mpi_reduce.hpp"

namespace kabalova_v_mpi_reduce {
std::vector<int> generateRandomVector(size_t size, int left, int right) {
  int seed = 100;
  std::mt19937 gen(seed);
  std::uniform_int_distribution distrib(left, right);
  std::vector result(size, 0);
  for (size_t i = 0; i < size; i++) {
    result[i] = distrib(gen);
  }
  return result;
}

int plus(std::vector<int>& vec) {
  int result = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    result += vec[i];
  }
  return result;
}

int multiply(std::vector<int>& vec) {
  int result = 1;
  for (size_t i = 0; i < vec.size(); i++) {
    result *= vec[i];
  }
  return result;
}

int land(std::vector<int>& vec) {
  bool result = true;
  for (size_t i = 0; i < vec.size(); i++) {
    result = result && static_cast<bool>(vec[i]);
  }
  return static_cast<int>(result);
}

int lor(std::vector<int>& vec) {
  bool result = false;
  for (size_t i = 0; i < vec.size(); i++) {
    result = result || static_cast<bool>(vec[i]);
  }
  return static_cast<int>(result);
}

int lxor(std::vector<int>& vec) {
  bool result = false;
  for (size_t i = 0; i < vec.size(); i++) {
    bool res1 = !static_cast<bool>(vec[i]);
    bool res2 = !result;
    result = res1 != res2;
  }
  return static_cast<int>(result);
}

int band(std::vector<int>& vec) {
  int result = 1;
  for (size_t i = 0; i < vec.size(); i++) {
    result = result & vec[i];
  }
  return result;
}

int bor(std::vector<int>& vec) {
  int result = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    result = result | vec[i];
  }
  return result;
}
int bxor(std::vector<int>& vec) {
  int result = 0;
  for (size_t i = 0; i < vec.size(); i++) {
    result = result ^ vec[i];
  }
  return result;
}
}  // namespace kabalova_v_mpi_reduce

TEST(kabalova_v_mpi_reduce, emptyVector) {
  boost::mpi::communicator world;
  std::vector<int> vec(0);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), false);
}

TEST(kabalova_v_mpi_reduce, sizeOneVector) {
  boost::mpi::communicator world;
  std::vector<int> vec(1);
  vec[0] = 1;

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(1, global_out[0]);
  }
}

TEST(kabalova_v_mpi_reduce, randomVecPlus) {
  boost::mpi::communicator world;
  size_t vecSize = 100;
  std::vector<int> vec = kabalova_v_mpi_reduce::generateRandomVector(vecSize, 0, 100);
  int answer = kabalova_v_mpi_reduce::plus(vec);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "+");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(answer, global_out[0]);
  }
}

TEST(kabalova_v_mpi_reduce, randomVecProd) {
  boost::mpi::communicator world;
  size_t vecSize = 10;
  std::vector<int> vec = kabalova_v_mpi_reduce::generateRandomVector(vecSize, 0, 10);
  int answer = kabalova_v_mpi_reduce::multiply(vec);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "*");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(answer, global_out[0]);
  }
}

TEST(kabalova_v_mpi_reduce, randomVecMax) {
  boost::mpi::communicator world;
  size_t vecSize = 100;
  std::vector<int> vec = kabalova_v_mpi_reduce::generateRandomVector(vecSize, -1000, 1000);
  int answer = *std::max_element(vec.begin(), vec.begin() + vecSize);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "max");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(answer, global_out[0]);
  }
}

TEST(kabalova_v_mpi_reduce, randomVecMin) {
  boost::mpi::communicator world;
  size_t vecSize = 100;
  std::vector<int> vec = kabalova_v_mpi_reduce::generateRandomVector(vecSize, -1000, 1000);
  int answer = *std::min_element(vec.begin(), vec.begin() + vecSize);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "min");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(answer, global_out[0]);
  }
}

TEST(kabalova_v_mpi_reduce, randomVecLand) {
  boost::mpi::communicator world;
  size_t vecSize = 100;
  std::vector<int> vec = kabalova_v_mpi_reduce::generateRandomVector(vecSize, 0, 100);
  int answer = kabalova_v_mpi_reduce::land(vec);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "&&");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(answer, global_out[0]);
  }
}

TEST(kabalova_v_mpi_reduce, randomVecLor) {
  boost::mpi::communicator world;
  size_t vecSize = 100;
  std::vector<int> vec = kabalova_v_mpi_reduce::generateRandomVector(vecSize, 0, 100);
  int answer = kabalova_v_mpi_reduce::lor(vec);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "||");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(answer, global_out[0]);
  }
}

TEST(kabalova_v_mpi_reduce, randomVecLxor) {
  boost::mpi::communicator world;
  size_t vecSize = 100;
  std::vector<int> vec = kabalova_v_mpi_reduce::generateRandomVector(vecSize, 0, 100);
  int answer = kabalova_v_mpi_reduce::lxor(vec);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "lxor");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(answer, global_out[0]);
  }
}

TEST(kabalova_v_mpi_reduce, randomVecBand) {
  boost::mpi::communicator world;
  size_t vecSize = 100;
  std::vector<int> vec = kabalova_v_mpi_reduce::generateRandomVector(vecSize, 0, 100);
  int answer = kabalova_v_mpi_reduce::band(vec);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "&");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(answer, global_out[0]);
  }
}

TEST(kabalova_v_mpi_reduce, randomVecBor) {
  boost::mpi::communicator world;
  size_t vecSize = 100;
  std::vector<int> vec = kabalova_v_mpi_reduce::generateRandomVector(vecSize, 0, 100);
  int answer = kabalova_v_mpi_reduce::bor(vec);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "|");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(answer, global_out[0]);
  }
}

TEST(kabalova_v_mpi_reduce, randomVecBxor) {
  boost::mpi::communicator world;
  size_t vecSize = 100;
  std::vector<int> vec = kabalova_v_mpi_reduce::generateRandomVector(vecSize, 0, 100);
  int answer = kabalova_v_mpi_reduce::bxor(vec);

  // Create data
  std::vector<int> global_out(1, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(vec.data()));
    taskDataMpi->inputs_count.emplace_back(vec.size());
    taskDataMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataMpi->outputs_count.emplace_back(global_out.size());
  }
  // Create Task
  kabalova_v_mpi_reduce::TestMPITaskParallel testMpiTaskParallel(taskDataMpi, "^");
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(answer, global_out[0]);
  }
}