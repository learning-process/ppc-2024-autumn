#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <vector>

#include "mpi/komshina_d_sort_radius_for_real_numbers_with_simple_merge/include/ops_mpi.hpp"

namespace mpi = boost::mpi;

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, Test_Empty_Vector) {
  mpi::environment env;
  mpi::communicator world;

  std::vector<double> empty_vec;
  std::vector<double> sorted_empty_vec(empty_vec.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(empty_vec.data()));
    taskDataPar->inputs_count.emplace_back(empty_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_empty_vec.data()));
    taskDataPar->outputs_count.emplace_back(sorted_empty_vec.size());
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(taskDataPar);
  ASSERT_TRUE(parallelSortTask.validation()) << "Validation failed!";
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_TRUE(sorted_empty_vec.empty()) << "Empty vector was not handled correctly!";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, Test_Single_Element_Vector) {
  mpi::environment env;
  mpi::communicator world;

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
  ASSERT_TRUE(parallelSortTask.validation()) << "Validation failed!";
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(sorted_single_vec, single_vec) << "Single element vector was not sorted correctly!";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, Test_All_Equal_Elements) {
  mpi::environment env;
  mpi::communicator world;

  std::vector<double> equal_vec(1000, 42.0);
  std::vector<double> sorted_equal_vec(equal_vec.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(equal_vec.data()));
    taskDataPar->inputs_count.emplace_back(equal_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_equal_vec.data()));
    taskDataPar->outputs_count.emplace_back(sorted_equal_vec.size());
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(taskDataPar);
  ASSERT_TRUE(parallelSortTask.validation()) << "Validation failed!";
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(sorted_equal_vec, equal_vec) << "Equal elements vector was not sorted correctly!";
  }
}

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, Test_Mixed_Positive_Negative) {
  mpi::environment env;
  mpi::communicator world;

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
  ASSERT_TRUE(parallelSortTask.validation()) << "Validation failed!";
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(sorted_mixed_vec, expected_sorted_vec)
        << "Mixed positive and negative numbers were not sorted correctly!";
  }
}