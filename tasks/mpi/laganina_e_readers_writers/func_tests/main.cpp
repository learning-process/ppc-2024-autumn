#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <vector>

#include "mpi/laganina_e_readers_writers/include/ops_mpi.hpp"

namespace laganina_e_readers_writers {

std::vector<int> getRandomVector(int sz, int a, int b) {
  std::random_device dev;
  std::mt19937 gen(dev());
  if (a > b) {
    throw std::out_of_range("range is incorrect");
  }
  std::uniform_int_distribution<> dist(a, b);

  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

}  // namespace laganina_e_readers_writers

TEST(laganina_e_readers_writers_mpi, test_getRandomVector) {
  std::vector<int> global_vec;
  ASSERT_ANY_THROW(global_vec = laganina_e_readers_writers::getRandomVector(1, 1001, 1000));
}

TEST(laganina_e_readers_writers_mpi, test_vector_10) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  const int count_size_vector = 10;
  int count = 0;  // this variable is needed to find out how many writers there were
  std::vector<int> global_vec;
  ASSERT_NO_THROW(global_vec = laganina_e_readers_writers::getRandomVector(count_size_vector, -1000, 1000));
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> exp_parallel = global_vec;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count));
    taskDataPar->outputs_count.emplace_back(out_vec.size());
  }

  laganina_e_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      exp_parallel[i] += count;
    }
    ASSERT_EQ(exp_parallel, out_vec);
  }
}

TEST(laganina_e_readers_writers_mpi, test_vector_20) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  const int count_size_vector = 20;
  int count = 0;  // this variable is needed to find out how many writers there were
  std::vector<int> global_vec;
  ASSERT_NO_THROW(global_vec = laganina_e_readers_writers::getRandomVector(count_size_vector, -1000, 1000));
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> exp_parallel = global_vec;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count));
    taskDataPar->outputs_count.emplace_back(out_vec.size());
  }

  laganina_e_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      exp_parallel[i] += count;
    }
    ASSERT_EQ(exp_parallel, out_vec);
  }
}

TEST(laganina_e_readers_writers_mpi, test_vector_32) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  const int count_size_vector = 32;
  int count = 0;  // this variable is needed to find out how many writers there were
  std::vector<int> global_vec;
  ASSERT_NO_THROW(global_vec = laganina_e_readers_writers::getRandomVector(count_size_vector, -1000, 1000));
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> exp_parallel = global_vec;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count));
    taskDataPar->outputs_count.emplace_back(out_vec.size());
  }

  laganina_e_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      exp_parallel[i] += count;
    }
    ASSERT_EQ(exp_parallel, out_vec);
  }
}

TEST(laganina_e_readers_writers_mpi, test_vector_128) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  const int count_size_vector = 128;
  int count = 0;  // this variable is needed to find out how many writers there were
  std::vector<int> global_vec;
  ASSERT_NO_THROW(global_vec = laganina_e_readers_writers::getRandomVector(count_size_vector, -1000, 1000));
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> exp_parallel = global_vec;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count));
    taskDataPar->outputs_count.emplace_back(out_vec.size());
  }

  laganina_e_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      exp_parallel[i] += count;
    }
    ASSERT_EQ(exp_parallel, out_vec);
  }
}

TEST(laganina_e_readers_writers_mpi, test_vector_100) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  const int count_size_vector = 100;
  int count = 0;  // this variable is needed to find out how many writers there were
  std::vector<int> global_vec;
  ASSERT_NO_THROW(global_vec = laganina_e_readers_writers::getRandomVector(count_size_vector, -1000, 1000));
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> exp_parallel = global_vec;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count));
    taskDataPar->outputs_count.emplace_back(out_vec.size());
  }

  laganina_e_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      exp_parallel[i] += count;
    }
    ASSERT_EQ(exp_parallel, out_vec);
  }
}

TEST(laganina_e_readers_writers_mpi, test_vector_500) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  const int count_size_vector = 500;
  int count = 0;  // this variable is needed to find out how many writers there were
  std::vector<int> global_vec;
  ASSERT_NO_THROW(global_vec = laganina_e_readers_writers::getRandomVector(count_size_vector, -1000, 1000));
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> exp_parallel = global_vec;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count));
    taskDataPar->outputs_count.emplace_back(out_vec.size());
  }

  laganina_e_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      exp_parallel[i] += count;
    }
    ASSERT_EQ(exp_parallel, out_vec);
  }
}

TEST(laganina_e_readers_writers_mpi, test_vector_999) {
  boost::mpi::communicator world;

  if (world.size() < 2) {
    GTEST_SKIP();
  }

  const int count_size_vector = 999;
  int count = 0;  // this variable is needed to find out how many writers there were
  std::vector<int> global_vec;
  ASSERT_NO_THROW(global_vec = laganina_e_readers_writers::getRandomVector(count_size_vector, -1000, 1000));
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> exp_parallel = global_vec;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&count));
    taskDataPar->outputs_count.emplace_back(out_vec.size());
  }

  laganina_e_readers_writers_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    for (int i = 0; i < count_size_vector; i++) {
      exp_parallel[i] += count;
    }
    ASSERT_EQ(exp_parallel, out_vec);
  }
}
