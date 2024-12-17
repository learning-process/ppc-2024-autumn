#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/mironov_a_quick_sort/include/ops_mpi.hpp"
using namespace std;
namespace mironov_a_quick_sort_mpi {

std::vector<int> get_random_vector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);

  const int mod = 1000000;
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % mod;
  }

  return vec;
}
}  // namespace mironov_a_quick_sort_mpi

TEST(mironov_a_quick_sort_mpi, Test_Sort_1) {
  boost::mpi::communicator world;
  // Create TaskData
  const int count = 20;
  std::vector<int> global_vec;
  std::vector<int> global_max;
  std::vector<int> gold;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData
    
    global_vec.resize(count);
    global_max.resize(count);
    for (int i = 0; i < count; ++i) {
      global_vec[i] = count - i;
    }
    gold = global_vec;
    sort(gold.begin(), gold.end());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  //cout << "Epoch 1 " << world.rank() << endl;
  mironov_a_quick_sort_mpi::QuickSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  //cout << "Epoch 2 " << world.rank() << endl;
  testMpiTaskParallel.pre_processing();
  //cout << "Epoch 3 " << world.rank() << endl;
  testMpiTaskParallel.run();
  //cout << "Epoch 4 " << world.rank() << endl;
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::vector<int32_t> reference_max(count);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max, gold);
    ASSERT_EQ(global_max, gold);
  }
}

TEST(mironov_a_quick_sort_mpi, Test_Sort_2) {
  boost::mpi::communicator world;
  // Create TaskData
  const int count = 30000;
  std::vector<int> global_vec;
  std::vector<int> global_max;
  std::vector<int> gold;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData

    global_vec = mironov_a_quick_sort_mpi::get_random_vector(count);
    global_max.resize(count);
    gold = global_vec;
    sort(gold.begin(), gold.end());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  // cout << "Epoch 1 " << world.rank() << endl;
  mironov_a_quick_sort_mpi::QuickSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  // cout << "Epoch 2 " << world.rank() << endl;
  testMpiTaskParallel.pre_processing();
  // cout << "Epoch 3 " << world.rank() << endl;
  testMpiTaskParallel.run();
  // cout << "Epoch 4 " << world.rank() << endl;
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::vector<int32_t> reference_max(count);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max, gold);
    ASSERT_EQ(global_max, gold);
  }
}

TEST(mironov_a_quick_sort_mpi, Test_Sort_3) {
  boost::mpi::communicator world;
  // Create TaskData
  const int count = 5000;
  std::vector<int> global_vec;
  std::vector<int> global_max;
  std::vector<int> gold;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData

    global_vec = mironov_a_quick_sort_mpi::get_random_vector(count);
    global_max.resize(count);
    gold = global_vec;
    sort(gold.begin(), gold.end());

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_max.data()));
    taskDataPar->outputs_count.emplace_back(global_max.size());
  }
  // cout << "Epoch 1 " << world.rank() << endl;
  mironov_a_quick_sort_mpi::QuickSortMPI testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  // cout << "Epoch 2 " << world.rank() << endl;
  testMpiTaskParallel.pre_processing();
  // cout << "Epoch 3 " << world.rank() << endl;
  testMpiTaskParallel.run();
  // cout << "Epoch 4 " << world.rank() << endl;
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::vector<int32_t> reference_max(count);

    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(reference_max.data()));
    taskDataSeq->outputs_count.emplace_back(reference_max.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(reference_max, gold);
    ASSERT_EQ(global_max, gold);
  }
}
