#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <random>

#include "mpi/petrov_o_radix_sort_with_simple_merge/include/ops_mpi.hpp"

using namespace petrov_o_radix_sort_with_simple_merge_mpi;

TEST(petrov_o_radix_sort_with_simple_merge_mpi, BasicSortTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  if (world.rank() == 0) {
    in = {8, 3};
    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation());
  ASSERT_TRUE(testTaskParallel.pre_processing());
  ASSERT_TRUE(testTaskParallel.run());
  ASSERT_TRUE(testTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out);
  }

  world.barrier();
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi, NegativeNumbersTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  if (world.rank() == 0) {
    in = {-100, -5, -3, 2, 7, 12};
    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation());
  ASSERT_TRUE(testTaskParallel.pre_processing());
  ASSERT_TRUE(testTaskParallel.run());
  ASSERT_TRUE(testTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out);
  }

  world.barrier();
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi, ReverseSortedTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  if (world.rank() == 0) {
    in = {10, 8, 6, 4, 2, 0};
    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation());
  ASSERT_TRUE(testTaskParallel.pre_processing());
  ASSERT_TRUE(testTaskParallel.run());
  ASSERT_TRUE(testTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out);
  }

  world.barrier();
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi, DuplicateElementsTest) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  if (world.rank() == 0) {
    in = {5, 5, 5, 5, 5, 5};
    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation());
  ASSERT_TRUE(testTaskParallel.pre_processing());
  ASSERT_TRUE(testTaskParallel.run());
  ASSERT_TRUE(testTaskParallel.post_processing());

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());
    ASSERT_EQ(expected, out);
  }

  world.barrier();
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi, RandomValuesTest_10n_size) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  if (world.rank() == 0) {
    const size_t array_size = 1000;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(gen);
    }

    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation()) << "Validation failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.pre_processing()) << "Pre-processing failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.run()) << "Run failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.post_processing()) << "Post-processing failed on rank " << world.rank();

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());

    ASSERT_EQ(expected, out) << "Sorted array does not match the expected result.";
  }

  world.barrier();
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi, RandomValuesTest_2n_size) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  if (world.rank() == 0) {
    const size_t array_size = 1024;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(gen);
    }

    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation()) << "Validation failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.pre_processing()) << "Pre-processing failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.run()) << "Run failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.post_processing()) << "Post-processing failed on rank " << world.rank();

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());

    ASSERT_EQ(expected, out) << "Sorted array does not match the expected result.";
  }

  world.barrier();
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi, RandomValuesTest_3n_size) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  if (world.rank() == 0) {
    const size_t array_size = 2187;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(gen);
    }

    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation()) << "Validation failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.pre_processing()) << "Pre-processing failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.run()) << "Run failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.post_processing()) << "Post-processing failed on rank " << world.rank();

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());

    ASSERT_EQ(expected, out) << "Sorted array does not match the expected result.";
  }

  world.barrier();
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi, RandomValuesTest_prime_values) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  if (world.rank() == 0) {
    const size_t array_size = 1013;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(-1000000, 1000000);

    in.resize(array_size);
    for (size_t i = 0; i < array_size; ++i) {
      in[i] = dist(gen);
    }

    out.resize(in.size(), 0);
  }

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    taskData->inputs_count.emplace_back(in.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  TaskParallel testTaskParallel(taskData);

  ASSERT_TRUE(testTaskParallel.validation()) << "Validation failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.pre_processing()) << "Pre-processing failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.run()) << "Run failed on rank " << world.rank();
  ASSERT_TRUE(testTaskParallel.post_processing()) << "Post-processing failed on rank " << world.rank();

  if (world.rank() == 0) {
    std::vector<int> expected = in;
    std::sort(expected.begin(), expected.end());

    ASSERT_EQ(expected, out) << "Sorted array does not match the expected result.";
  }

  world.barrier();
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, BasicSortTest) {
  std::vector<int> in{8, 3};
  std::vector<int> out(in.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  TaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, NegativeNumbersTest) {
  std::vector<int> in{-100, -5, -3, 2, 7, 12};
  std::vector<int> out(in.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  TaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, ReverseSortedTest) {
  std::vector<int> in{10, 8, 6, 4, 2, 0};
  std::vector<int> out(in.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  TaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, DuplicateElementsTest) {
  std::vector<int> in{5, 5, 5, 5, 5, 5};
  std::vector<int> out(in.size(), 0);

  auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  TaskSequential testTaskSequential(taskDataSeq);

  ASSERT_TRUE(testTaskSequential.validation());
  ASSERT_TRUE(testTaskSequential.pre_processing());
  ASSERT_TRUE(testTaskSequential.run());
  ASSERT_TRUE(testTaskSequential.post_processing());

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, out);
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, RandomValuesTest_10n_size) {
  std::vector<int> in;
  std::vector<int> out;

  const size_t array_size = 1000;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-1000000, 1000000);

  in.resize(array_size);
  for (size_t i = 0; i < array_size; ++i) {
    in[i] = dist(gen);
  }

  out.resize(in.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  TaskSequential testTaskSequential(taskData);

  ASSERT_TRUE(testTaskSequential.validation()) << "Validation failed";
  ASSERT_TRUE(testTaskSequential.pre_processing()) << "Pre-processing failed";
  ASSERT_TRUE(testTaskSequential.run()) << "Run failed";
  ASSERT_TRUE(testTaskSequential.post_processing()) << "Post-processing failed";

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());

  ASSERT_EQ(expected, out) << "Sorted array does not match the expected result.";
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, RandomValuesTest_2n_size) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> in;
  std::vector<int> out;
  const size_t array_size = 1024;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-1000000, 1000000);

  in.resize(array_size);
  for (size_t i = 0; i < array_size; ++i) {
    in[i] = dist(gen);
  }

  out.resize(in.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  TaskSequential testTaskSequential(taskData);

  ASSERT_TRUE(testTaskSequential.validation()) << "Validation failed";
  ASSERT_TRUE(testTaskSequential.pre_processing()) << "Pre-processing failed";
  ASSERT_TRUE(testTaskSequential.run()) << "Run failed";
  ASSERT_TRUE(testTaskSequential.post_processing()) << "Post-processing failed";

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());

  ASSERT_EQ(expected, out) << "Sorted array does not match the expected result.";
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, RandomValuesTest_3n_size) {
  std::vector<int> in;
  std::vector<int> out;
  const size_t array_size = 2187;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-1000000, 1000000);

  in.resize(array_size);
  for (size_t i = 0; i < array_size; ++i) {
    in[i] = dist(gen);
  }

  out.resize(in.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  TaskSequential testTaskSequential(taskData);

  ASSERT_TRUE(testTaskSequential.validation()) << "Validation failed";
  ASSERT_TRUE(testTaskSequential.pre_processing()) << "Pre-processing failed";
  ASSERT_TRUE(testTaskSequential.run()) << "Run failed";
  ASSERT_TRUE(testTaskSequential.post_processing()) << "Post-processing failed";

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());

  ASSERT_EQ(expected, out) << "Sorted array does not match the expected result.";
}

TEST(petrov_o_radix_sort_with_simple_merge_mpi_seq, RandomValuesTest_prime_values) {
  std::vector<int> in;
  std::vector<int> out;
  const size_t array_size = 1013;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(-1000000, 1000000);

  in.resize(array_size);
  for (size_t i = 0; i < array_size; ++i) {
    in[i] = dist(gen);
  }

  out.resize(in.size(), 0);

  auto taskData = std::make_shared<ppc::core::TaskData>();

  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  taskData->inputs_count.emplace_back(in.size());
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskData->outputs_count.emplace_back(out.size());

  TaskSequential testTaskSequential(taskData);

  ASSERT_TRUE(testTaskSequential.validation()) << "Validation failed";
  ASSERT_TRUE(testTaskSequential.pre_processing()) << "Pre-processing failed";
  ASSERT_TRUE(testTaskSequential.run()) << "Run failed";
  ASSERT_TRUE(testTaskSequential.post_processing()) << "Post-processing failed";

  std::vector<int> expected = in;
  std::sort(expected.begin(), expected.end());

  ASSERT_EQ(expected, out) << "Sorted array does not match the expected result.";
}