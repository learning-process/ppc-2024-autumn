#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include "mpi/korablev_v_quick_sort_simple_merge/include/ops_mpi.hpp"

namespace korablev_v_qucik_sort_simple_merge_mpi {
std::vector<int> generate_random_vector(size_t n, int min_val = -1000, int max_val = 1000) {
  std::vector<int> vec(n);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dist(min_val, max_val);

  for (size_t i = 0; i < n; ++i) {
    vec[i] = dist(gen);
  }

  return vec;
}
}  // namespace korablev_v_qucik_sort_simple_merge_mpi

void run_quick_sort_test_for_vector_size(size_t vector_size) {
  boost::mpi::communicator world;

  auto random_vector = korablev_v_qucik_sort_simple_merge_mpi::generate_random_vector(vector_size);

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

  korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeParallel parallel_sort(taskDataPar);
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

    korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeSequential sequential_sort(taskDataSeq);
    ASSERT_TRUE(sequential_sort.validation());
    sequential_sort.pre_processing();
    sequential_sort.run();
    sequential_sort.post_processing();
  }

  if (world.rank() == 0) {
    for (size_t i = 0; i < vector_size; ++i) {
      ASSERT_NEAR(parallel_result[i], sequential_result[i], 1e-9);
    }
  } else {
    ASSERT_TRUE(true) << "Process " << world.rank() << " completed successfully.";
  }
}

TEST(korablev_v_quick_sort_mpi, test_vector_10) { run_quick_sort_test_for_vector_size(10); }
TEST(korablev_v_quick_sort_mpi, test_vector_100) { run_quick_sort_test_for_vector_size(100); }
TEST(korablev_v_quick_sort_mpi, test_vector_1000) { run_quick_sort_test_for_vector_size(1000); }
TEST(korablev_v_quick_sort_mpi, test_vector_5000) { run_quick_sort_test_for_vector_size(5000); }
TEST(korablev_v_quick_sort_mpi, test_vector_10000) { run_quick_sort_test_for_vector_size(10000); }

TEST(korablev_v_quick_sort_mpi, debug_test) {
  boost::mpi::communicator world;
  const size_t array_size = 6;
  std::vector<size_t> in_size(1, array_size);
  std::vector<int> input_data = {5, 3, 8, 6, 2, 7};
  std::vector<int> expected_output = {2, 3, 5, 6, 7, 8};
  std::vector<int> out(array_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
  taskDataPar->inputs_count.emplace_back(in_size.size());
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_data.data()));
  taskDataPar->inputs_count.emplace_back(input_data.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataPar->outputs_count.emplace_back(out.size());

  korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeParallel quickSortTask(taskDataPar);

  ASSERT_TRUE(quickSortTask.validation());

  quickSortTask.pre_processing();
  quickSortTask.run();
  quickSortTask.post_processing();

  if (world.rank() == 0) {
    for (size_t i = 0; i < array_size; ++i) {
      ASSERT_EQ(out[i], expected_output[i]);
    }
  }
}