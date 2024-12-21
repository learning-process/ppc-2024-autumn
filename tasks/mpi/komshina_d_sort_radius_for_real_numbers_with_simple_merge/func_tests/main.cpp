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

TEST(komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi, Test_Large_Vector) {
  mpi::environment env;
  mpi::communicator world;

  std::vector<double> global_vec(1000000, 0.0);
  for (int i = 0; i < 1000000; ++i) {
    global_vec[i] = rand() % 1000;
  }
  std::vector<double> global_sorted_vec(global_vec.size(), 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sorted_vec.data()));
    taskDataPar->outputs_count.emplace_back(global_sorted_vec.size());
  }

  komshina_d_sort_radius_for_real_numbers_with_simple_merge_mpi::TestMPITaskParallel parallelSortTask(taskDataPar);
  ASSERT_TRUE(parallelSortTask.validation()) << "Validation failed!";
  parallelSortTask.pre_processing();
  parallelSortTask.run();
  parallelSortTask.post_processing();

  if (world.rank() == 0) {
    std::sort(global_vec.begin(), global_vec.end());
    ASSERT_EQ(global_sorted_vec, global_vec) << "Large vector sort result mismatch!";
  }
}