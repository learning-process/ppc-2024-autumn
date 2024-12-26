#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/mironov_a_quick_sort/include/ops_mpi.hpp"
using namespace std;
namespace mironov_a_quick_sort_mpi {

std::vector<int> get_random_vector(int sz, int min, int max) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);

  int mod = max - min + 1;
  if (mod < 0) {
    mod *= -1;
  }
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % mod + min;
  }

  return vec;
}
}  // namespace mironov_a_quick_sort_mpi

TEST(mironov_a_quick_sort_mpi, Test_Sort_1) {
  boost::mpi::communicator world;
  // Create TaskData
  const int count = 20;
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> gold;
  std::shared_ptr<ppc::core::TaskData> ParallelData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData

    in.resize(count);
    out.resize(count);
    for (int i = 0; i < count; ++i) {
      in[i] = count - i;
    }
    gold = in;
    sort(gold.begin(), gold.end());

    ParallelData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    ParallelData->inputs_count.emplace_back(in.size());
    ParallelData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    ParallelData->outputs_count.emplace_back(out.size());
  }
  mironov_a_quick_sort_mpi::QuickSortMPI parallelTask(ParallelData);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::vector<int32_t> out_ref(count);

    std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
    seqTask->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    seqTask->inputs_count.emplace_back(in.size());
    seqTask->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_ref.data()));
    seqTask->outputs_count.emplace_back(out_ref.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(seqTask);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_ref, gold);
    ASSERT_EQ(out, gold);
  }
}

TEST(mironov_a_quick_sort_mpi, Test_Sort_2) {
  boost::mpi::communicator world;
  // Create TaskData
  const int count = 30000;
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> gold;
  std::shared_ptr<ppc::core::TaskData> ParallelData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData

    in = mironov_a_quick_sort_mpi::get_random_vector(count, 1, 10000000);
    out.resize(count);
    gold = in;
    sort(gold.begin(), gold.end());

    ParallelData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    ParallelData->inputs_count.emplace_back(in.size());
    ParallelData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    ParallelData->outputs_count.emplace_back(out.size());
  }
  mironov_a_quick_sort_mpi::QuickSortMPI parallelTask(ParallelData);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::vector<int32_t> out_ref(count);

    std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
    seqTask->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    seqTask->inputs_count.emplace_back(in.size());
    seqTask->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_ref.data()));
    seqTask->outputs_count.emplace_back(out_ref.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(seqTask);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_ref, gold);
    ASSERT_EQ(out, gold);
  }
}

TEST(mironov_a_quick_sort_mpi, Test_Sort_3) {
  boost::mpi::communicator world;
  // Create TaskData
  const int count = 5000;
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> gold;
  std::shared_ptr<ppc::core::TaskData> ParallelData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData

    in = mironov_a_quick_sort_mpi::get_random_vector(count, 1, 2);
    out.resize(count);
    gold = in;
    sort(gold.begin(), gold.end());

    ParallelData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    ParallelData->inputs_count.emplace_back(in.size());
    ParallelData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    ParallelData->outputs_count.emplace_back(out.size());
  }

  mironov_a_quick_sort_mpi::QuickSortMPI parallelTask(ParallelData);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::vector<int32_t> out_ref(count);

    std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
    seqTask->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    seqTask->inputs_count.emplace_back(in.size());
    seqTask->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_ref.data()));
    seqTask->outputs_count.emplace_back(out_ref.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(seqTask);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_ref, gold);
    ASSERT_EQ(out, gold);
  }
}

TEST(mironov_a_quick_sort_mpi, Test_Sort_4) {
  boost::mpi::communicator world;
  // Create TaskData
  const int count = 1024;
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> gold;
  std::shared_ptr<ppc::core::TaskData> ParallelData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData

    in = mironov_a_quick_sort_mpi::get_random_vector(count, -100, 200);
    out.resize(count);
    gold = in;
    sort(gold.begin(), gold.end());

    ParallelData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    ParallelData->inputs_count.emplace_back(in.size());
    ParallelData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    ParallelData->outputs_count.emplace_back(out.size());
  }

  mironov_a_quick_sort_mpi::QuickSortMPI parallelTask(ParallelData);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::vector<int32_t> out_ref(count);

    std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
    seqTask->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    seqTask->inputs_count.emplace_back(in.size());
    seqTask->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_ref.data()));
    seqTask->outputs_count.emplace_back(out_ref.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(seqTask);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_ref, gold);
    ASSERT_EQ(out, gold);
  }
}

TEST(mironov_a_quick_sort_mpi, Test_Sort_5) {
  boost::mpi::communicator world;
  // Create TaskData
  const int count = 10;
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> gold;
  std::shared_ptr<ppc::core::TaskData> ParallelData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData

    in = mironov_a_quick_sort_mpi::get_random_vector(count, -100, -10);
    out.resize(count);
    gold = in;
    sort(gold.begin(), gold.end());

    ParallelData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    ParallelData->inputs_count.emplace_back(in.size());
    ParallelData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    ParallelData->outputs_count.emplace_back(out.size());
  }

  mironov_a_quick_sort_mpi::QuickSortMPI parallelTask(ParallelData);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::vector<int32_t> out_ref(count);

    std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
    seqTask->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    seqTask->inputs_count.emplace_back(in.size());
    seqTask->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_ref.data()));
    seqTask->outputs_count.emplace_back(out_ref.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(seqTask);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_ref, gold);
    ASSERT_EQ(out, gold);
  }
}

TEST(mironov_a_quick_sort_mpi, Test_Sort_6) {
  boost::mpi::communicator world;
  // Create TaskData
  const int count = 1;
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> gold;
  std::shared_ptr<ppc::core::TaskData> ParallelData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData

    in = mironov_a_quick_sort_mpi::get_random_vector(count, 0, std::numeric_limits<int>::max());
    out.resize(count);
    gold = in;
    sort(gold.begin(), gold.end());

    ParallelData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    ParallelData->inputs_count.emplace_back(in.size());
    ParallelData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    ParallelData->outputs_count.emplace_back(out.size());
  }

  mironov_a_quick_sort_mpi::QuickSortMPI parallelTask(ParallelData);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::vector<int32_t> out_ref(count);

    std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
    seqTask->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    seqTask->inputs_count.emplace_back(in.size());
    seqTask->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_ref.data()));
    seqTask->outputs_count.emplace_back(out_ref.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(seqTask);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_ref, gold);
    ASSERT_EQ(out, gold);
  }
}

TEST(mironov_a_quick_sort_mpi, Test_Sort_reversed_array) {
  boost::mpi::communicator world;
  // Create TaskData
  const int count = 1;
  std::vector<int> in;
  std::vector<int> out;
  std::vector<int> gold;
  std::shared_ptr<ppc::core::TaskData> ParallelData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData

    in = mironov_a_quick_sort_mpi::get_random_vector(count, 0, std::numeric_limits<int>::max());
    out.resize(count);
    gold = in;
    sort(gold.begin(), gold.end());
    sort(in.rbegin(), in.rend());

    ParallelData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    ParallelData->inputs_count.emplace_back(in.size());
    ParallelData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    ParallelData->outputs_count.emplace_back(out.size());
  }

  mironov_a_quick_sort_mpi::QuickSortMPI parallelTask(ParallelData);
  ASSERT_EQ(parallelTask.validation(), true);
  parallelTask.pre_processing();
  parallelTask.run();
  parallelTask.post_processing();

  if (world.rank() == 0) {
    // Create TaskData
    std::vector<int32_t> out_ref(count);

    std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
    seqTask->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    seqTask->inputs_count.emplace_back(in.size());
    seqTask->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_ref.data()));
    seqTask->outputs_count.emplace_back(out_ref.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(seqTask);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_ref, gold);
    ASSERT_EQ(out, gold);
  }
}

TEST(mironov_a_quick_sort_mpi, Test_wrong_input) {
  boost::mpi::communicator world;
  // Create TaskData

  std::vector<int> in;
  std::vector<int> out;

  std::shared_ptr<ppc::core::TaskData> ParallelData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    // Create TaskData

    ParallelData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    ParallelData->inputs_count.emplace_back(in.size());
    ParallelData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    ParallelData->outputs_count.emplace_back(out.size());
    mironov_a_quick_sort_mpi::QuickSortMPI parallelTask(ParallelData);
    ASSERT_EQ(parallelTask.validation(), false);

    // Create TaskData
    std::vector<int32_t> out_ref;

    std::shared_ptr<ppc::core::TaskData> seqTask = std::make_shared<ppc::core::TaskData>();
    seqTask->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    seqTask->inputs_count.emplace_back(in.size());
    seqTask->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_ref.data()));
    seqTask->outputs_count.emplace_back(out_ref.size());

    // Create Task
    mironov_a_quick_sort_mpi::QuickSortSequential testMpiTaskSequential(seqTask);
    ASSERT_EQ(testMpiTaskSequential.validation(), false);
  }
}
