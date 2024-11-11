#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/polikanov_v_max_of_vector_elements/include/ops_mpi.hpp"
namespace polikanov_v {
std::vector<int> getRandomVector(int sz, int lower, int upper) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = lower + gen() % (upper - lower + 1);
  }
  vec[0] = upper;
  return vec;
}
}  // namespace polikanov_v

TEST(polikanov_v_max_of_vector_elements_MPI, Test_Valid_false) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(100, 1);
  std::vector<int32_t> global_sum(2, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    global_vec = polikanov_v::getRandomVector(100, 0, 2);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  polikanov_v_max_of_vector_elements::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  polikanov_v_max_of_vector_elements::TestMPITaskSequential testMpiTaskSequential(taskDataPar);
  if (world.rank() == 0) {
    ASSERT_FALSE(testMpiTaskParallel.validation());
    ASSERT_FALSE(testMpiTaskSequential.validation());
  }
}

TEST(polikanov_v_max_of_vector_elements_MPI, Test_Empty_Array) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> ans(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    taskDataPar->outputs_count.emplace_back(ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<polikanov_v_max_of_vector_elements::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(INT_MIN, ans[0]);
  }
}

TEST(polikanov_v_max_of_vector_elements_MPI, Test_Negative_Numbers) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> ans(1, 0);
  int n = 10;
  int lower = -100;
  int upper = -1;
  int max_el = upper;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = polikanov_v::getRandomVector(n, lower, upper);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    taskDataPar->outputs_count.emplace_back(ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<polikanov_v_max_of_vector_elements::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(max_el, ans[0]);
  }
}

TEST(polikanov_v_max_of_vector_elements_MPI, Test_Mixed_Numbers) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> ans(1, 0);
  int n = 10;
  int lower = -50;
  int upper = 50;
  int max_el = 50;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = polikanov_v::getRandomVector(n, lower, upper);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    taskDataPar->outputs_count.emplace_back(ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<polikanov_v_max_of_vector_elements::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(max_el, ans[0]);
  }
}

TEST(polikanov_v_max_of_vector_elements_MPI, Test_Valid_true) {
  boost::mpi::communicator world;
  std::vector<int> global_vec(100, 1);
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }
  polikanov_v_max_of_vector_elements::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_TRUE(testMpiTaskParallel.validation());
}
TEST(polikanov_v_max_of_vector_elements_MPI, Test_Main) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> ans(1, 0);
  int n = 100;
  int lower = 0;
  int max_el = 100;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = polikanov_v::getRandomVector(n, lower, max_el);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    taskDataPar->outputs_count.emplace_back(ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<polikanov_v_max_of_vector_elements::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(max_el, ans[0]);
  }
}

TEST(polikanov_v_max_of_vector_elements_MPI, Test_Main1) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> ans(1, 0);
  int n = 10;
  int lower = 0;
  int max_el = 100;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    global_vec = polikanov_v::getRandomVector(n, lower, max_el);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans.data()));
    taskDataPar->outputs_count.emplace_back(ans.size());
  }

  auto testMpiTaskParallel = std::make_shared<polikanov_v_max_of_vector_elements::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(max_el, ans[0]);
  }
}
TEST(polikanov_v_max_of_vector_elements_MPI, Test_Main2) {
  std::vector<int> global_v_Seq;
  std::vector<int32_t> ansSeq(1, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_v_Seq = {10, 2, 3, 4, 5, 6};
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_v_Seq.data()));
  taskDataSeq->inputs_count.emplace_back(global_v_Seq.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansSeq.data()));
  taskDataSeq->outputs_count.emplace_back(ansSeq.size());
  auto testMpiTaskSequential = std::make_shared<polikanov_v_max_of_vector_elements::TestMPITaskSequential>(taskDataSeq);
  ASSERT_EQ(testMpiTaskSequential->validation(), true);
  testMpiTaskSequential->pre_processing();
  testMpiTaskSequential->run();
  testMpiTaskSequential->post_processing();
  ASSERT_EQ(10, ansSeq[0]);
}
TEST(polikanov_v_max_of_vector_elements_MPI, Test_Main3) {
  boost::mpi::communicator world;
  std::vector<int> global_v_Par;

  std::vector<int32_t> ansPar(1, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  global_v_Par = {10, 2, 3, 4, 5, 6};

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_v_Par.data()));
    taskDataPar->inputs_count.emplace_back(global_v_Par.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ansPar.data()));
    taskDataPar->outputs_count.emplace_back(ansPar.size());
  }

  auto testMpiTaskParallel = std::make_shared<polikanov_v_max_of_vector_elements::TestMPITaskParallel>(taskDataPar);

  ASSERT_EQ(testMpiTaskParallel->validation(), true);

  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  if (world.rank() == 0) {
    ASSERT_EQ(10, ansPar[0]);
  }
}
