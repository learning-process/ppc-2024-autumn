// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/vladimirova_j_gather/include/ops_mpi.hpp"
#include "mpi/vladimirova_j_gather/include/ops_mpi_not_my_gather.hpp"
using namespace vladimirova_j_gather_mpi;
using namespace vladimirova_j_not_my_gather_mpi;

namespace vladimirova_j_gather_mpi {

std::vector<int> getRandomVal(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  vec[0] = 2;
  vec[sz - 1] = 2;
  for (int i = 1; i < sz - 1; i++) {
    if ((i != 0) && (vec[i - 1] != 2)) {
      vec[i] = 2;
      continue;
    }
    vec[i] = (gen() % 3 - 1);
    if (vec[i] == 0) vec[i] = 2;
  }
  return vec;
};
}  // namespace vladimirova_j_gather_mpi

TEST(Parallel_Operations_MPI, vladimirova_j_gather_test_validation_zero) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
    vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_gather_test_validation) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {2, 2, -1, 2, 2, 2, 2, 2, -1, 2, 5, 2, -1, 2, 2, 2, -1, -1, 2,
                                    2, 2, 1,  2, 2, 2, 1, 2, 2,  2, 2, 2, 1,  2, 2, 2, 2,  1,  2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
    vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_gather_1_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {2, 2, -1, 2, 2, 2, 2, 2, -1, 2, 2, 2, -1, 2, 2, 2, -1, -1, 2,
                                    2, 2, 1,  2, 2, 2, 1, 2, 2,  2, 2, 2, 1,  2, 2, 2, 2,  1,  2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  }

  vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ((int)taskDataPar->outputs_count[0], 6);
    ASSERT_EQ(ans_buf_vec, ans_vec);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_gather_forward_backward_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {2, 2, 2, 2, 2, 2, 2, -1, -1, 2, 2, 2, 2, 2, 2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {2, -1, -1};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  }

  vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ((size_t)taskDataPar->outputs_count[0], ans_vec.size());
    ASSERT_EQ(ans_buf_vec, ans_vec);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_gather_right_left_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {-1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  }

  vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ((size_t)taskDataPar->outputs_count[0], ans_vec.size());

    ASSERT_EQ(ans_buf_vec, ans_vec);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_gather_more_dead_ends_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {1, 2, 2, 1, 2, 1,  -1, -1, -1, 2, 1, 2, 2, 2,  1,  2, 1,  2,
                                    2, 1, 2, 1, 2, -1, -1, 2,  -1, 2, 1, 1, 2, -1, -1, 2, -1, 2};
  // 1 2 2    1 -1 -1 1    2   2 2 1 2 1 2 2 1 2 1 2 -1 -1 2 1 1 -1 2 1 2
  std::vector<int32_t> ans_vec = {1, 2, 2, 1, -1, -1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, -1, -1, 1, -1, -1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  }

  vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.validation();
  // ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ((size_t)taskDataPar->outputs_count[0], ans_vec.size());
    ASSERT_EQ(ans_buf_vec, ans_vec);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_random_test) {
  boost::mpi::communicator world;
  std::vector<int> some_dead_end;
  std::vector<int> tmp;
  std::vector<int> global_vector;
  //{0,1,2,3,4,5,6,7,8,9};

  int noDEnd = 0;
  for (int j = 0; j < 10; j++) {
    some_dead_end = vladimirova_j_gather_mpi::getRandomVal(5);
    tmp = vladimirova_j_gather_mpi::getRandomVal(15);
    noDEnd += 15;
    global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
    global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
    global_vector.push_back(-1);
    global_vector.push_back(-1);
    noDEnd += 2;
    for (int i = some_dead_end.size() - 1; i >= 0; i--) {
      if (some_dead_end[i] != 2) {
        global_vector.push_back(-1 * some_dead_end[i]);
      } else {
        global_vector.push_back(2);
      }
    }
  }
  std::vector<int32_t> ans_buf_vec(noDEnd);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  }

  vladimirova_j_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ((int)taskDataPar->outputs_count[0] <= noDEnd, true);
  }
}

TEST(Parallel_Operations_MPI, vladimirova_j_not_my_gather_test_validation_zero) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(0);
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
    vladimirova_j_not_my_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_not_my_gather_test_validation) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {2, 2, -1, 2, 2, 2, 2, 2, -1, 2, 5, 2, -1, 2, 2, 2, -1, -1, 2,
                                    2, 2, 1,  2, 2, 2, 1, 2, 2,  2, 2, 2, 1,  2, 2, 2, 2,  1,  2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
    vladimirova_j_not_my_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
    ASSERT_EQ(testMpiTaskParallel.validation(), false);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_not_my_gather_1_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {2, 2, -1, 2, 2, 2, 2, 2, -1, 2, 2, 2, -1, 2, 2, 2, -1, -1, 2,
                                    2, 2, 1,  2, 2, 2, 1, 2, 2,  2, 2, 2, 1,  2, 2, 2, 2,  1,  2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  }

  vladimirova_j_not_my_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ((int)taskDataPar->outputs_count[0], 6);

    ASSERT_EQ(ans_buf_vec, ans_vec);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_not_my_gather_forward_backward_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {2, 2, 2, 2, 2, 2, 2, -1, -1, 2, 2, 2, 2, 2, 2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {2, -1, -1};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  }

  vladimirova_j_not_my_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ((size_t)taskDataPar->outputs_count[0], ans_vec.size());

    ASSERT_EQ(ans_buf_vec, ans_vec);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_not_my_gather_right_left_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {-1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  }

  vladimirova_j_not_my_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ((size_t)taskDataPar->outputs_count[0], ans_vec.size());

    ASSERT_EQ(ans_buf_vec, ans_vec);
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_not_my_gather_more_dead_ends_test) {
  boost::mpi::communicator world;
  std::vector<int> global_vector = {1, 2, 2, 1, 2, 1,  -1, -1, -1, 2, 1, 2, 2, 2,  1,  2, 1,  2,
                                    2, 1, 2, 1, 2, -1, -1, 2,  -1, 2, 1, 1, 2, -1, -1, 2, -1, 2};
  // 1 2 2    1 -1 -1 1    2   2 2 1 2 1 2 2 1 2 1 2 -1 -1 2 1 1 -1 2 1 2
  std::vector<int32_t> ans_vec = {1, 2, 2, 1, -1, -1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, -1, -1, 1, -1, -1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  }

  vladimirova_j_not_my_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  testMpiTaskParallel.validation();
  // ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(ans_buf_vec, ans_vec);

    ASSERT_EQ((size_t)taskDataPar->outputs_count[0], ans_vec.size());
  }
}
TEST(Parallel_Operations_MPI, vladimirova_j_not_my_gather_random_test) {
  boost::mpi::communicator world;
  std::vector<int> some_dead_end;
  std::vector<int> tmp;
  std::vector<int> global_vector;
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};

  int noDEnd = 0;
  for (int j = 0; j < 10; j++) {
    some_dead_end = vladimirova_j_gather_mpi::getRandomVal(5);
    tmp = vladimirova_j_gather_mpi::getRandomVal(15);
    noDEnd += 15;
    global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
    global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
    global_vector.push_back(-1);
    global_vector.push_back(-1);
    noDEnd += 2;
    for (int i = some_dead_end.size() - 1; i >= 0; i--) {
      if (some_dead_end[i] != 2)
        global_vector.push_back(-1 * some_dead_end[i]);
      else
        global_vector.push_back(2);
    }
  }

  std::vector<int32_t> ans_buf_vec(noDEnd);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  }

  vladimirova_j_not_my_gather_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ((int)taskDataPar->outputs_count[0] <= noDEnd, true);
  }
}

TEST(Sequential_Operations_MPI, vladimirova_j_test_validation_zero) {
  std::vector<int> global_vector = {};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(0);
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), false);
}
TEST(Sequential_Operations_MPI, vladimirova_j_test_validation) {
  std::vector<int> global_vector = {2, 2, -1, 2, 2, 2, 2, 2, -1, 2, 5, 2, -1, 2, 2, 2, -1, -1, 2,
                                    2, 2, 1,  2, 2, 2, 1, 2, 2,  2, 2, 2, 1,  2, 2, 2, 2,  1,  2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), false);
}
TEST(Sequential_Operations_MPI, vladimirova_j_forward_backward_test) {
  std::vector<int> global_vector = {2, 2, 2, 2, 2, 2, 2, -1, -1, 2, 2, 2, 2, 2, 2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {2, -1, -1};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  ASSERT_EQ((size_t)taskDataSeq->outputs_count[0], ans_vec.size());
  ASSERT_EQ(ans_buf_vec, ans_vec);
}
TEST(Sequential_Operations_MPI, vladimirova_j_gather_right_left_test) {
  std::vector<int> global_vector = {-1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  ASSERT_EQ((size_t)taskDataSeq->outputs_count[0], ans_vec.size());

  ASSERT_EQ(ans_buf_vec, ans_vec);
}
TEST(Sequential_Operations_MPI, vladimirova_j_gather_more_dead_ends_test) {
  std::vector<int> global_vector = {1, 2, 2, 1, 2, 1,  -1, -1, -1, 2, 1, 2, 2, 2,  1,  2, 1,  2,
                                    2, 1, 2, 1, 2, -1, -1, 2,  -1, 2, 1, 1, 2, -1, -1, 2, -1, 2};
  // 1 2 2    1 -1 -1 1    2   2 2 1 2 1 2 2 1 2 1 2 -1 -1 2 1 1 -1 2 1 2
  std::vector<int32_t> ans_vec = {1, 2, 2, 1, -1, -1, 1, 2, 2, 2, 1, 2, 1, 2, 2, 1, -1, -1, 1, -1, -1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));
  vladimirova_j_gather_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);

  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();

  ASSERT_EQ((size_t)taskDataSeq->outputs_count[0], ans_vec.size());

  ASSERT_EQ(ans_buf_vec, ans_vec);
}
TEST(Sequential_Operations_MPI, vladimirova_j_random_test) {
  std::vector<int> some_dead_end;
  std::vector<int> tmp;
  std::vector<int> global_vector;
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};

  int noDEnd = 0;
  for (int j = 0; j < 10; j++) {
    some_dead_end = vladimirova_j_gather_mpi::getRandomVal(5);
    tmp = vladimirova_j_gather_mpi::getRandomVal(15);
    noDEnd += 15;
    global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
    global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
    global_vector.push_back(-1);
    global_vector.push_back(-1);
    noDEnd += 2;
    for (int i = some_dead_end.size() - 1; i >= 0; i--) {
      if (some_dead_end[i] != 2)
        global_vector.push_back(-1 * some_dead_end[i]);
      else
        global_vector.push_back(2);
    }
  }

  std::vector<int32_t> ans_buf_vec(noDEnd);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential.validation(), true);
  testMpiTaskSequential.pre_processing();
  testMpiTaskSequential.run();
  testMpiTaskSequential.post_processing();
  ASSERT_EQ((int)taskDataSeq->outputs_count[0] <= noDEnd, true);
}
TEST(Sequential_Operations_MPI, vladimirova_j_not_gather_1_test) {
  std::vector<int> global_vector = {2, 2, -1, 2, 2, 2, 2, 2, -1, 2, 2, 2, -1, 2, 2, 2, -1, -1, 2,
                                    2, 2, 1,  2, 2, 2, 1, 2, 2,  2, 2, 2, 1,  2, 2, 2, 2,  1,  2};
  //{0,1,2,3,4,5,6,7,8,9};
  std::vector<int32_t> ans_vec = {-1, -1, 2, 2, 1, 2};
  std::vector<int32_t> ans_buf_vec(ans_vec.size());

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(ans_buf_vec.data()));

  vladimirova_j_gather_mpi::TestMPITaskSequential testMPITaskSequential(taskDataSeq);
  ASSERT_EQ(testMPITaskSequential.validation(), true);
  testMPITaskSequential.pre_processing();
  testMPITaskSequential.run();
  testMPITaskSequential.post_processing();

  ASSERT_EQ((size_t)taskDataSeq->outputs_count[0], ans_vec.size());
  ASSERT_EQ(ans_buf_vec, ans_vec);
}
