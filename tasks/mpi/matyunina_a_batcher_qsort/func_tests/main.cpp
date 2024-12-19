// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <climits>
#include <random>
#include <vector>

#include "mpi/matyunina_a_batcher_qsort/include/ops_mpi.hpp"

namespace matyunina_a_batcher_qsort_mpi {

std::vector<int32_t> generateRandomVector(size_t size, int32_t min_value, int32_t max_value) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int32_t> dist(min_value, max_value);

  std::vector<int32_t> in(size);
  for (size_t i = 0; i < size; i++) {
    in[i] = dist(gen);
  }

  return in;
}

}  // namespace matyunina_a_batcher_qsort_mpi

TEST(matyunina_a_batcher_qsort_mpi, random_vector) {
  boost::mpi::communicator world;
  int32_t size = 10;
  int32_t min = -500;
  int32_t max = 500;
  // Create data
  std::vector<int32_t> out;
  std::vector<int32_t> in;
  std::vector<int32_t> sorted;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = matyunina_a_batcher_qsort_mpi::generateRandomVector(size, min, max);
    sorted = in;
    out.resize(in.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }
  // Create Task
  matyunina_a_batcher_qsort_mpi::TestTaskParallel<int32_t> testTaskParallel(taskData);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> out_seq(in.size());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskData->outputs_count.emplace_back(out.size());

    // Create Task
    matyunina_a_batcher_qsort_mpi::TestTaskSequential<int32_t> testTaskSequential(taskData);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    std::sort(sorted.begin(), sorted.end());
    ASSERT_EQ(sorted, out);
    ASSERT_EQ(sorted, out_seq);
  }
}

TEST(matyunina_a_batcher_qsort_mpi, zero) {
  boost::mpi::communicator world;
  std::vector<int32_t> in;
  std::vector<int32_t> out;

  std::vector<int32_t> sorted(in);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create Task
  matyunina_a_batcher_qsort_mpi::TestTaskParallel<int32_t> testTaskParallel(taskData);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> out_seq(in.size());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskData->outputs_count.emplace_back(out.size());

    // Create Task
    matyunina_a_batcher_qsort_mpi::TestTaskSequential<int32_t> testTaskSequential(taskData);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    std::sort(sorted.begin(), sorted.end());
    ASSERT_EQ(sorted, out);
    ASSERT_EQ(sorted, out_seq);
  }
}

TEST(matyunina_a_batcher_qsort_mpi, single) {
  boost::mpi::communicator world;

  std::vector<int32_t> in = {42};
  std::vector<int32_t> out = {42};
  std::vector<int32_t> sorted(in);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create Task
  matyunina_a_batcher_qsort_mpi::TestTaskParallel<int32_t> testTaskParallel(taskData);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> out_seq(in.size());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskData->outputs_count.emplace_back(out.size());

    // Create Task
    matyunina_a_batcher_qsort_mpi::TestTaskSequential<int32_t> testTaskSequential(taskData);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    std::sort(sorted.begin(), sorted.end());
    ASSERT_EQ(sorted, out);
    ASSERT_EQ(sorted, out_seq);
  }
}

TEST(matyunina_a_batcher_qsort_mpi, duplicated_elements) {
  boost::mpi::communicator world;

  std::vector<int32_t> in = {3, 4, 4, 1, 5, 5, 2, 6, 5, 3};
  std::vector<int32_t> out(in.size());
  std::vector<int32_t> sorted(in);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create Task
  matyunina_a_batcher_qsort_mpi::TestTaskParallel<int32_t> testTaskParallel(taskData);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> out_seq(in.size());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskData->outputs_count.emplace_back(out.size());

    // Create Task
    matyunina_a_batcher_qsort_mpi::TestTaskSequential<int32_t> testTaskSequential(taskData);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    std::sort(sorted.begin(), sorted.end());
    ASSERT_EQ(sorted, out);
    ASSERT_EQ(sorted, out_seq);
  }
}

TEST(matyunina_a_batcher_qsort_mpi, video_example) {
  boost::mpi::communicator world;

  std::vector<int32_t> in = {8, 2, 5, 10, 1, 7, 3, 12, 6, 11, 4, 9};
  std::vector<int32_t> out(in.size());
  std::vector<int32_t> sorted(in);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create Task
  matyunina_a_batcher_qsort_mpi::TestTaskParallel<int32_t> testTaskParallel(taskData);
  ASSERT_EQ(testTaskParallel.validation(), true);
  testTaskParallel.pre_processing();
  testTaskParallel.run();
  testTaskParallel.post_processing();

  if (world.rank() == 0) {
    std::vector<int32_t> out_seq(in.size());

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_seq.data()));
    taskData->outputs_count.emplace_back(out.size());

    // Create Task
    matyunina_a_batcher_qsort_mpi::TestTaskSequential<int32_t> testTaskSequential(taskData);
    ASSERT_EQ(testTaskSequential.validation(), true);
    testTaskSequential.pre_processing();
    testTaskSequential.run();
    testTaskSequential.post_processing();

    std::sort(sorted.begin(), sorted.end());
    ASSERT_EQ(sorted, out);
    ASSERT_EQ(sorted, out_seq);
  }
}