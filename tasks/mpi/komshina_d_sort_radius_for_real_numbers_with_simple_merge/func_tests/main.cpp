#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <vector>

#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, Test_Single_Element_Vector) {
  boost::mpi::communicator world;

  std::vector<double> single_vec = {42.0};
  std::vector<double> sorted_single_vec(single_vec.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(single_vec.data()));
    taskDataPar->inputs_count.emplace_back(single_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_single_vec.data()));
    taskDataPar->outputs_count.emplace_back(sorted_single_vec.size());
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(taskDataPar);
  ASSERT_TRUE(parallelSortTask.validation()) << "Parallel validation failed!";
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(sorted_single_vec, single_vec) << "Parallel: Single element vector was not sorted correctly!";
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(single_vec.data()));
    taskDataSeq->inputs_count.emplace_back(single_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_single_vec.data()));
    taskDataSeq->outputs_count.emplace_back(sorted_single_vec.size());
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential sequentialSortTask(taskDataSeq);
  ASSERT_TRUE(sequentialSortTask.validation()) << "Sequential validation failed!";
  sequentialSortTask.pre_processing();
  sequentialSortTask.run();
  sequentialSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(sorted_single_vec, single_vec) << "Sequential: Single element vector was not sorted correctly!";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, Test_Mixed_Positive_Negative) {
  boost::mpi::communicator world;

  std::vector<double> mixed_vec = {-3.14, 2.71, -42.0, 0.0, 7.5, -1.23};
  std::vector<double> expected_sorted_vec = {-42.0, -3.14, -1.23, 0.0, 2.71, 7.5};
  std::vector<double> sorted_mixed_vec(mixed_vec.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(mixed_vec.data()));
    taskDataPar->inputs_count.emplace_back(mixed_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_mixed_vec.data()));
    taskDataPar->outputs_count.emplace_back(sorted_mixed_vec.size());
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(taskDataPar);
  ASSERT_TRUE(parallelSortTask.validation()) << "Parallel validation failed!";
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(sorted_mixed_vec, expected_sorted_vec)
        << "Parallel: Mixed positive and negative numbers were not sorted correctly!";
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(mixed_vec.data()));
    taskDataSeq->inputs_count.emplace_back(mixed_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_mixed_vec.data()));
    taskDataSeq->outputs_count.emplace_back(sorted_mixed_vec.size());
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential sequentialSortTask(taskDataSeq);
  ASSERT_TRUE(sequentialSortTask.validation()) << "Sequential validation failed!";
  sequentialSortTask.pre_processing();
  sequentialSortTask.run();
  sequentialSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(sorted_mixed_vec, expected_sorted_vec)
        << "Sequential: Mixed positive and negative numbers were not sorted correctly!";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, Test_Duplicate_Elements) {
  boost::mpi::communicator world;

  std::vector<double> duplicate_vec = {5.5, 2.2, 5.5, 3.3, 2.2};
  std::vector<double> expected_sorted_vec = {2.2, 2.2, 3.3, 5.5, 5.5};
  std::vector<double> sorted_duplicate_vec(duplicate_vec.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(duplicate_vec.data()));
    taskDataPar->inputs_count.emplace_back(duplicate_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_duplicate_vec.data()));
    taskDataPar->outputs_count.emplace_back(sorted_duplicate_vec.size());
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(taskDataPar);
  ASSERT_TRUE(parallelSortTask.validation()) << "Parallel validation failed!";
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(sorted_duplicate_vec, expected_sorted_vec)
        << "Parallel: Vector with duplicate elements was not sorted correctly!";
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(duplicate_vec.data()));
    taskDataSeq->inputs_count.emplace_back(duplicate_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_duplicate_vec.data()));
    taskDataSeq->outputs_count.emplace_back(sorted_duplicate_vec.size());
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskSequential sequentialSortTask(taskDataSeq);
  ASSERT_TRUE(sequentialSortTask.validation()) << "Sequential validation failed!";
  sequentialSortTask.pre_processing();
  sequentialSortTask.run();
  sequentialSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(sorted_duplicate_vec, expected_sorted_vec)
        << "Sequential: Vector with duplicate elements was not sorted correctly!";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, Test_Empty_Vector) {
  boost::mpi::communicator world;

  std::vector<double> empty_vec = {};
  std::vector<double> sorted_empty_vec(empty_vec.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(empty_vec.data()));
    taskDataPar->inputs_count.emplace_back(empty_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_empty_vec.data()));
    taskDataPar->outputs_count.emplace_back(sorted_empty_vec.size());
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(taskDataPar);
  ASSERT_TRUE(parallelSortTask.validation()) << "Parallel validation failed!";
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(sorted_empty_vec, empty_vec) << "Parallel: Empty vector was not sorted correctly!";
  }
}