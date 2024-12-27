#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <random>
#include <vector>

#include "mpi/belov_a_radix_sort_with_batcher_mergesort/include/ops_mpi.hpp"

namespace belov_a_radix_batcher_mergesort_mpi {
vector<bigint> generate_mixed_values_array(int n) {
  random_device rd;
  mt19937 gen(rd());

  uniform_int_distribution<bigint> small_range(-999LL, 999LL);
  uniform_int_distribution<bigint> medium_range(-10000LL, 10000LL);
  uniform_int_distribution<bigint> large_range(-10000000000LL, 10000000000LL);
  uniform_int_distribution<int> choice(0, 2);

  vector<bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    if (choice(gen) == 0) {
      arr.push_back(small_range(gen));
    } else if (choice(gen) == 1) {
      arr.push_back(medium_range(gen));
    } else {
      arr.push_back(large_range(gen));
    }
  }
  return arr;
}

vector<bigint> generate_int_array(int n) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dist(numeric_limits<int>::min(), numeric_limits<int>::max());

  vector<bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    arr.push_back(dist(gen));
  }
  return arr;
}

vector<bigint> generate_bigint_array(int n) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<bigint> dist(numeric_limits<bigint>::min() / 2, numeric_limits<bigint>::max() / 2);

  vector<bigint> arr;
  arr.reserve(n);

  for (int i = 0; i < n; ++i) {
    arr.push_back(dist(gen));
  }
  return arr;
}
}  // namespace belov_a_radix_batcher_mergesort_mpi

using namespace belov_a_radix_batcher_mergesort_mpi;

TEST(belov_a_radix_batcher_mergesort_mpi, test_random_small_mixedV_vector) {
  boost::mpi::communicator world;

  int n = 1024;
  vector<bigint> arr = generate_mixed_values_array(n);

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    RadixBatcherMergesortSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    EXPECT_EQ(arr, solutionSeq);
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_random_medium_mixedV_vector) {
  boost::mpi::communicator world;

  int n = 8192;
  vector<bigint> arr = generate_mixed_values_array(n);

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    RadixBatcherMergesortSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    EXPECT_EQ(arr, solutionSeq);
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_random_large_mixedV_vector) {
  boost::mpi::communicator world;

  int n = 65536;
  vector<bigint> arr = generate_mixed_values_array(n);

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    RadixBatcherMergesortSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    EXPECT_EQ(arr, solutionSeq);
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_random_small_bigintV_vector) {
  boost::mpi::communicator world;

  int n = 1024;
  vector<bigint> arr = generate_bigint_array(n);

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    RadixBatcherMergesortSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    EXPECT_EQ(arr, solutionSeq);
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_random_medium_bigintV_vector) {
  boost::mpi::communicator world;

  int n = 8192;
  vector<bigint> arr = generate_bigint_array(n);

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    RadixBatcherMergesortSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    EXPECT_EQ(arr, solutionSeq);
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_random_large_bigintV_vector) {
  boost::mpi::communicator world;

  int n = 32768;
  vector<bigint> arr = generate_bigint_array(n);

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    RadixBatcherMergesortSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    EXPECT_EQ(arr, solutionSeq);
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_predefined_intV_vector) {
  boost::mpi::communicator world;

  int n = 13;
  vector<bigint> arr = {-2147483648, 2147483647, -1000, 1001, 0, 124, -123, 42, -43, 2, -1, -999999, 999998};

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    RadixBatcherMergesortSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    EXPECT_EQ(arr, solutionSeq);
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_negative_values_only) {
  boost::mpi::communicator world;

  int n = 16;
  vector<bigint> arr = {-5799507787,   -640070325,  -9553275362,  -6878351,     -8042130384, -8443056136,
                        -2713654083,   -5467368930, -72261888,    -50054111267, -883621556,  -2780342973,
                        -291923548343, -5485582439, -94518015487, -3833280574};

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    RadixBatcherMergesortSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    EXPECT_EQ(arr, solutionSeq);
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_positive_values_only) {
  boost::mpi::communicator world;

  int n = 9;
  vector<bigint> arr = {7666729877,  20598172788, 7934137,    289,    627274651110,
                        28942526990, 742923702,   3438922384, 7936973};

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel.validation(), true);
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    RadixBatcherMergesortSequential testMpiTaskSeq(taskDataSeq);
    ASSERT_TRUE(testMpiTaskSeq.validation());
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    EXPECT_EQ(arr, solutionSeq);
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_one_element_input_array) {
  boost::mpi::communicator world;

  int n = 1;
  vector<bigint> arr = {64};

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());
  }

  RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
  EXPECT_TRUE(testMpiTaskParallel.validation());
  testMpiTaskParallel.pre_processing();
  testMpiTaskParallel.run();
  testMpiTaskParallel.post_processing();

  if (world.rank() == 0) {
    shared_ptr<ppc::core::TaskData> taskDataSeq = make_shared<ppc::core::TaskData>();
    vector<bigint> solutionSeq(n);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataSeq->inputs_count.emplace_back(arr.size());
    taskDataSeq->inputs_count.emplace_back(n);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(solutionSeq.data()));
    taskDataSeq->outputs_count.emplace_back(solutionSeq.size());

    RadixBatcherMergesortSequential testMpiTaskSeq(taskDataSeq);
    EXPECT_TRUE(testMpiTaskSeq.validation());
    testMpiTaskSeq.pre_processing();
    testMpiTaskSeq.run();
    testMpiTaskSeq.post_processing();

    EXPECT_EQ(arr, solutionSeq);
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_empty_input_validation_condition1) {
  boost::mpi::communicator world;

  // zero length, empty input
  int n = 0;
  vector<bigint> arr = {};

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());

    RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_empty_input_validation_condition2) {
  boost::mpi::communicator world;

  // declared length is 5, but empty input
  int n = 5;
  vector<bigint> arr = {};

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());

    RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}

TEST(belov_a_radix_batcher_mergesort_mpi, test_validation_condition3) {
  boost::mpi::communicator world;

  // declared length is not equal to real one
  int n = 8;
  vector<bigint> arr = {62584567, 0, -1953346};

  shared_ptr<ppc::core::TaskData> taskDataPar = make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->inputs_count.emplace_back(arr.size());
    taskDataPar->inputs_count.emplace_back(n);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(arr.data()));
    taskDataPar->outputs_count.emplace_back(arr.size());

    RadixBatcherMergesortParallel testMpiTaskParallel(taskDataPar);
    EXPECT_FALSE(testMpiTaskParallel.validation());
  }
}