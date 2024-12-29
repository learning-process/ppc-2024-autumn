#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "mpi/korablev_v_quick_sort_simple_merge/include/ops_mpi.hpp"

namespace korablev_v_quick_sort_simple_merge_mpi {
std::vector<int> generate_random_vector(size_t n, int min_val = -1000, int max_val = 1000) {
  std::vector<int> vec(n);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(min_val, max_val);
  for (size_t i = 0; i < n; ++i) {
    vec[i] = dist(gen);
  }
  std::sort(vec.begin(), vec.end(), std::greater<>());
  return vec;
}
}  // namespace korablev_v_quick_sort_simple_merge_mpi

void run_quick_sort_test_for_vector_size(size_t vector_size) {
  boost::mpi::communicator world;

  auto random_vector = korablev_v_quick_sort_simple_merge_mpi::generate_random_vector(vector_size);

  std::vector<int> parallel_result(vector_size, 0.0);
  std::vector<int> sequential_result(vector_size, 0.0);

  size_t vector_size_copy = vector_size;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_size_copy));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_vector.data()));
    taskDataPar->inputs_count.emplace_back(random_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(parallel_result.data()));
    taskDataPar->outputs_count.emplace_back(parallel_result.size());
  }

  korablev_v_quick_sort_simple_merge_mpi::QuickSortSimpleMergeParallel parallel_sort(taskDataPar);
  ASSERT_TRUE(parallel_sort.validation());
  parallel_sort.pre_processing();
  parallel_sort.run();
  parallel_sort.post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&vector_size_copy));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_vector.data()));
    taskDataSeq->inputs_count.emplace_back(random_vector.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sequential_result.data()));
    taskDataSeq->outputs_count.emplace_back(sequential_result.size());

    korablev_v_quick_sort_simple_merge_mpi::QuickSortSimpleMergeSequential sequential_sort(taskDataSeq);
    ASSERT_TRUE(sequential_sort.validation());
    sequential_sort.pre_processing();
    sequential_sort.run();
    sequential_sort.post_processing();

    EXPECT_EQ(parallel_result, sequential_result);
  }
}

TEST(korablev_v_quick_sort_mpi, test_empty_vector) { run_quick_sort_test_for_vector_size(0); }
TEST(korablev_v_quick_sort_mpi, test_single_element) { run_quick_sort_test_for_vector_size(1); }
TEST(korablev_v_quick_sort_mpi, test_vector_2) { run_quick_sort_test_for_vector_size(2); }
TEST(korablev_v_quick_sort_mpi, test_vector_3) { run_quick_sort_test_for_vector_size(3); }
TEST(korablev_v_quick_sort_mpi, test_vector_4) { run_quick_sort_test_for_vector_size(4); }
TEST(korablev_v_quick_sort_mpi, test_vector_5) { run_quick_sort_test_for_vector_size(5); }
TEST(korablev_v_quick_sort_mpi, test_vector_7) { run_quick_sort_test_for_vector_size(7); }
TEST(korablev_v_quick_sort_mpi, test_vector_8) { run_quick_sort_test_for_vector_size(8); }
TEST(korablev_v_quick_sort_mpi, test_vector_16) { run_quick_sort_test_for_vector_size(16); }
TEST(korablev_v_quick_sort_mpi, test_vector_32) { run_quick_sort_test_for_vector_size(32); }
TEST(korablev_v_quick_sort_mpi, test_vector_64) { run_quick_sort_test_for_vector_size(64); }
TEST(korablev_v_quick_sort_mpi, test_vector_128) { run_quick_sort_test_for_vector_size(128); }
TEST(korablev_v_quick_sort_mpi, test_vector_256) { run_quick_sort_test_for_vector_size(256); }
TEST(korablev_v_quick_sort_mpi, test_vector_1024) { run_quick_sort_test_for_vector_size(1024); }
TEST(korablev_v_quick_sort_mpi, test_vector_2048) { run_quick_sort_test_for_vector_size(2048); }
TEST(korablev_v_quick_sort_mpi, test_vector_4096) { run_quick_sort_test_for_vector_size(4096); }
TEST(korablev_v_quick_sort_mpi, test_vector_10) { run_quick_sort_test_for_vector_size(10); }
TEST(korablev_v_quick_sort_mpi, test_vector_100) { run_quick_sort_test_for_vector_size(100); }
TEST(korablev_v_quick_sort_mpi, test_vector_1000) { run_quick_sort_test_for_vector_size(1000); }
TEST(korablev_v_quick_sort_mpi, test_vector_5000) { run_quick_sort_test_for_vector_size(5000); }
TEST(korablev_v_quick_sort_mpi, test_vector_10000) { run_quick_sort_test_for_vector_size(10000); }

TEST(korablev_v_quick_sort_mpi, test_negative_and_positive) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    const size_t array_size = 5;
    std::vector<size_t> in_size(1, array_size);
    std::vector<int> input_data = {-3, 2, -1, 0, 1};
    std::vector<int> expected_output = {-3, -1, 0, 1, 2};
    std::vector<int> out(array_size, 0);

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());

    korablev_v_quick_sort_simple_merge_mpi::QuickSortSimpleMergeParallel quickSortTask(taskDataPar);
    EXPECT_TRUE(quickSortTask.validation());
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(korablev_v_quick_sort_mpi, invalid_input_count) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    std::vector<size_t> in_size(1, 6);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());

    std::vector<int> out(6, 0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());

    korablev_v_quick_sort_simple_merge_mpi::QuickSortSimpleMergeParallel quickSortTask(taskDataPar);
    EXPECT_FALSE(quickSortTask.validation());
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(korablev_v_quick_sort_mpi, invalid_output_count) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    std::vector<size_t> in_size(1, 6);
    std::vector<int> input_data = {5, 3, 8, 6, 2, 7};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    korablev_v_quick_sort_simple_merge_mpi::QuickSortSimpleMergeParallel quickSortTask(taskDataPar);
    EXPECT_FALSE(quickSortTask.validation());
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(korablev_v_quick_sort_mpi, invalid_negative_size) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    auto invalid_size = static_cast<size_t>(-1);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&invalid_size));
    taskDataPar->inputs_count.emplace_back(1);

    std::vector<int> input_data = {5, 3, 8, 6, 2, 7};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    std::vector<int> out(6, 0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());

    korablev_v_quick_sort_simple_merge_mpi::QuickSortSimpleMergeParallel quickSortTask(taskDataPar);
    EXPECT_FALSE(quickSortTask.validation());
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(korablev_v_quick_sort_mpi, invalid_size_data_mismatch) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    size_t array_size = 6;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&array_size));
    taskDataPar->inputs_count.emplace_back(1);

    std::vector<int> input_data = {5, 3, 8};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    std::vector<int> out(6, 0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());

    korablev_v_quick_sort_simple_merge_mpi::QuickSortSimpleMergeParallel quickSortTask(taskDataPar);
    EXPECT_FALSE(quickSortTask.validation());
  } else {
    EXPECT_TRUE(true);
  }
}

TEST(korablev_v_quick_sort_mpi, invalid_output_size) {
  boost::mpi::communicator world;

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
    size_t array_size = 6;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&array_size));
    taskDataPar->inputs_count.emplace_back(1);

    std::vector<int> input_data = {5, 3, 8, 6, 2, 7};
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
    taskDataPar->inputs_count.emplace_back(input_data.size());

    std::vector<int> out(3, 0);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());

    korablev_v_quick_sort_simple_merge_mpi::QuickSortSimpleMergeParallel quickSortTask(taskDataPar);
    EXPECT_FALSE(quickSortTask.validation());
  } else {
    EXPECT_TRUE(true);
  }
}