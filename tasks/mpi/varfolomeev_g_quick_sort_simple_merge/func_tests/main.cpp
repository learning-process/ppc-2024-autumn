// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/varfolomeev_g_quick_sort_simple_merge/include/ops_mpi.hpp"

namespace varfolomeev_g_quick_sort_simple_merge_mpi {
std::vector<int> getRandomVector_mpi(int sz, int a, int b) {  // [a, b]
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % (b - a + 1) + a;
  }
  return vec;
}

std::vector<int> getAntisorted_mpi(int sz, int a) {  // [a + sz, a)
  std::vector<int> vec(sz);
  for (int i = a + sz, j = 0; i > a && j < sz; i--, j++) {
    vec[j] = i;
  }
  return vec;
}
}  // namespace varfolomeev_g_quick_sort_simple_merge_mpi

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_Antisorted_Positive) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 150;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getAntisorted_mpi(size, 200);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_Antisorted_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 150;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getAntisorted_mpi(size, -200);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_Antisorted_Positive_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 150;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getAntisorted_mpi(size, -75);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_RandomVector_Positive) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 150;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getRandomVector_mpi(size, 0, 200);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_RandomVector_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 150;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getRandomVector_mpi(size, -200, 0);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_RandomVector_Positive_Negative) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 150;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getRandomVector_mpi(size, -100, 100);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_Antisorted_64) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 64;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getAntisorted_mpi(size, 200);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_Antisorted_128) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 128;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getAntisorted_mpi(size, 200);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_Antisorted_512) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 512;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getAntisorted_mpi(size, 200);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_Antisorted_1024) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 1024;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getAntisorted_mpi(size, 200);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_Antisorted_4096) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 4096;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getAntisorted_mpi(size, 200);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_Antisorted_8192) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 8192;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getAntisorted_mpi(size, 200);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_EQ(global_res, check_vec);
  }
}

TEST(varfolomeev_g_quick_sort_simple_merge_mpi, Test_Antisorted_10000) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_res(global_vec.size());
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size = 10000;
    global_vec = varfolomeev_g_quick_sort_simple_merge_mpi::getAntisorted_mpi(size, 200);
    global_res.resize(global_vec.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  varfolomeev_g_quick_sort_simple_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> check_vec(global_res.size());
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(check_vec.data()));
    taskDataSeq->outputs_count.emplace_back(check_vec.size());

    varfolomeev_g_quick_sort_simple_merge_mpi::TestTaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    EXPECT_EQ(global_res, check_vec);
  }
}