// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/zaytsev_d_num_of_alternations_signs/include/ops_mpi.hpp"

TEST(zaytsev_d_num_of_alternations_signs_mpi, AllPositive) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_count[0], 0);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, AllNegative) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_count[0], 0);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, AlternatingPositiveNegative) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_count[0], 9);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, WithZeros) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {1, 0, -1, 0, 1, 0, -1, 0};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_count[0], 4);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, BigVector) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, -1, -1};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_count[0], 3);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, SmallVector) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {1, -1};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_count[0], 1);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, EmptyVector) {
  boost::mpi::communicator world;
  std::vector<int> test_vector = {};
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(test_vector.data()));
    taskDataPar->inputs_count.emplace_back(test_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(global_count[0], 0);
  }
}

TEST(zaytsev_d_num_of_alternations_signs_mpi, WithRandomVector) {
  boost::mpi::communicator world;
  int vector_size = 100;
  std::vector<int> random_vector = zaytsev_d_num_of_alternations_signs_mpi::getRandomVector(vector_size);
  std::vector<int32_t> global_count(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_vector.data()));
    taskDataPar->inputs_count.emplace_back(random_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_count.data()));
    taskDataPar->outputs_count.emplace_back(global_count.size());
  }

  zaytsev_d_num_of_alternations_signs_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    int expected_count = 0;
    for (size_t i = 1; i < random_vector.size(); i++) {
      if ((random_vector[i - 1] >= 0 && random_vector[i] < 0) || (random_vector[i - 1] < 0 && random_vector[i] >= 0)) {
        expected_count++;
      }
    }
    ASSERT_EQ(global_count[0], expected_count);
  }
}
