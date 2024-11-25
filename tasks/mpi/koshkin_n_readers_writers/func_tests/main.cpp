#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/koshkin_n_readers_writers/include/ops_mpi.hpp"

TEST(readers_writers_MPI, test_small_vector) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    ASSERT_TRUE(true);
    return;
  }

  const int count_size_vector = 15;

  std::vector<int> global_vec = koshkin_n_readers_writers_mpi::getRandomVector(count_size_vector);
  std::vector<int> exit_vec(count_size_vector, 0);
  std::vector<int> res_exp_paral = global_vec;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int num_writers = 0;
    for (int rank = 1; rank < world.size(); rank++) {
      if (rank % 2 != 0) num_writers++;
    }
    for (int i = 0; i < count_size_vector; i++) {
      res_exp_paral[i] += num_writers * 100;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(exit_vec.data()));
    taskDataPar->outputs_count.emplace_back(exit_vec.size());
  }

  koshkin_n_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(res_exp_paral, exit_vec);
  }
}

TEST(readers_writers_MPI, test_medium_vector) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    ASSERT_TRUE(true);
    return;
  }

  const int count_size_vector = 300;

  std::vector<int> global_vec = koshkin_n_readers_writers_mpi::getRandomVector(count_size_vector);
  std::vector<int> exit_vec(count_size_vector, 0);
  std::vector<int> res_exp_paral = global_vec;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int num_writers = 0;
    for (int rank = 1; rank < world.size(); rank++) {
      if (rank % 2 != 0) num_writers++;
    }
    for (int i = 0; i < count_size_vector; i++) {
      res_exp_paral[i] += num_writers * 100;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(exit_vec.data()));
    taskDataPar->outputs_count.emplace_back(exit_vec.size());
  }

  koshkin_n_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(res_exp_paral, exit_vec);
  }
}

TEST(readers_writers_MPI, test_large_vector) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    ASSERT_TRUE(true);
    return;
  }

  const int count_size_vector = 999;

  std::vector<int> global_vec = koshkin_n_readers_writers_mpi::getRandomVector(count_size_vector);
  std::vector<int> exit_vec(count_size_vector, 0);
  std::vector<int> res_exp_paral = global_vec;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int num_writers = 0;
    for (int rank = 1; rank < world.size(); rank++) {
      if (rank % 2 != 0) num_writers++;
    }
    for (int i = 0; i < count_size_vector; i++) {
      res_exp_paral[i] += num_writers * 100;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(exit_vec.data()));
    taskDataPar->outputs_count.emplace_back(exit_vec.size());
  }

  koshkin_n_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(res_exp_paral, exit_vec);
  }
}

TEST(readers_writers_MPI, test_empty_vector) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    ASSERT_TRUE(true);
    return;
  }

  const int count_size_vector = 0;

  std::vector<int> global_vec = {};
  std::vector<int> exit_vec(count_size_vector, 0);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(exit_vec.data()));
    taskDataPar->outputs_count.emplace_back(exit_vec.size());
    koshkin_n_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}