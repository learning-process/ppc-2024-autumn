#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>
#include <vector>

#include "../include/ops_mpi.hpp"

static std::vector<int> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = static_cast<int>(gen()) % 100;
  }
  return vec;
}

TEST(moiseev_a_radix_merge, test_sum) {
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 133;
    in = getRandomVector(count_size_vector);
    out.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  moiseev_a_radix_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> ref(in.size());

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataSeq->inputs_count.emplace_back(in.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(ref.data()));
    taskDataSeq->outputs_count.emplace_back(ref.size());

    moiseev_a_radix_merge_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out, ref);
  }
}

TEST(moiseev_a_radix_merge, test_compare_with_std_sort) {
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 200;
    in = getRandomVector(count_size_vector);
    out.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  moiseev_a_radix_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> ref = in;
    std::sort(ref.begin(), ref.end());

    ASSERT_EQ(out, ref);
  }
}

TEST(moiseev_a_radix_merge, test_small_vectors) {
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 10;
    in = getRandomVector(count_size_vector);
    out.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  moiseev_a_radix_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> ref = in;
    std::sort(ref.begin(), ref.end());

    ASSERT_EQ(out, ref);
  }
}

TEST(moiseev_a_radix_merge, test_empty_vector) {
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 0;
    in = getRandomVector(count_size_vector);
    out.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  moiseev_a_radix_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> ref = in;
    std::sort(ref.begin(), ref.end());

    ASSERT_EQ(out, ref);
  }
}

TEST(moiseev_a_radix_merge, test_compare_with_std_sort_large) {
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 300;
    in = getRandomVector(count_size_vector);
    out.resize(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  moiseev_a_radix_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> ref = in;
    std::sort(ref.begin(), ref.end());

    ASSERT_EQ(out, ref);
  }
}

TEST(moiseev_a_radix_merge, test_fixed_values) {
  boost::mpi::communicator world;

  std::vector<int> in = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5};
  std::vector<int> out;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    out.resize(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  moiseev_a_radix_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> ref = in;
    std::sort(ref.begin(), ref.end());

    ASSERT_EQ(out, ref);
  }
}

TEST(moiseev_a_radix_merge, test_negative_values) {
  boost::mpi::communicator world;

  std::vector<int> in = {-3, -1, -4, -1, -5};
  std::vector<int> out;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    out.resize(in.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  moiseev_a_radix_merge_mpi::TestMPITaskParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int> ref = in;
    std::sort(ref.begin(), ref.end());

    ASSERT_EQ(out, ref);
  }
}