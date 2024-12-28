// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/suvorov_d_shell_with_ord_merge/include/ops_mpi.hpp"

namespace suvorov_d_shell_with_ord_merge_mpi {
std::vector<int> get_rand_vector(const size_t vec_size, const int min_int = -10000, const int max_int = 10000) {
  std::random_device r_dev;
  std::mt19937 gen(r_dev());
  std::uniform_int_distribution<> distr(min_int, max_int);

  std::vector<int> new_vec(vec_size);
  std::generate(new_vec.begin(), new_vec.end(), [&]() { return distr(gen); });
  std::sort(new_vec.begin(), new_vec.end(), std::greater<>());
  return new_vec;
}
}  // namespace suvorov_d_shell_with_ord_merge_mpi

TEST(suvorov_d_shell_with_ord_merge_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> data_to_sort;
  size_t count_of_elems;
  std::vector<int> sorted_result_mpi;

  std::shared_ptr<ppc::core::TaskData> taskDataForSortingMpi = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    count_of_elems = 10000000;
    data_to_sort = suvorov_d_shell_with_ord_merge_mpi::get_rand_vector(count_of_elems);
    sorted_result_mpi.assign(count_of_elems, 0);
    taskDataForSortingMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_to_sort.data()));
    taskDataForSortingMpi->inputs_count.emplace_back(data_to_sort.size());
    taskDataForSortingMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_result_mpi.data()));
    taskDataForSortingMpi->outputs_count.emplace_back(sorted_result_mpi.size());
  }

  auto ShellSortMpi =
      std::make_shared<suvorov_d_shell_with_ord_merge_mpi::TaskShellSortParallel>(taskDataForSortingMpi);
  ASSERT_EQ(ShellSortMpi->validation(), true);
  ShellSortMpi->pre_processing();
  ShellSortMpi->run();
  ShellSortMpi->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ShellSortMpi);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_TRUE(std::is_sorted(sorted_result_mpi.begin(), sorted_result_mpi.end()));
  }
}

TEST(suvorov_d_shell_with_ord_merge_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> data_to_sort;
  size_t count_of_elems;
  std::vector<int> sorted_result_mpi;

  std::shared_ptr<ppc::core::TaskData> taskDataForSortingMpi = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    count_of_elems = 10000000;
    data_to_sort = suvorov_d_shell_with_ord_merge_mpi::get_rand_vector(count_of_elems);
    sorted_result_mpi.assign(count_of_elems, 0);
    taskDataForSortingMpi->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_to_sort.data()));
    taskDataForSortingMpi->inputs_count.emplace_back(data_to_sort.size());
    taskDataForSortingMpi->outputs.emplace_back(reinterpret_cast<uint8_t*>(sorted_result_mpi.data()));
    taskDataForSortingMpi->outputs_count.emplace_back(sorted_result_mpi.size());
  }

  auto ShellSortMpi =
      std::make_shared<suvorov_d_shell_with_ord_merge_mpi::TaskShellSortParallel>(taskDataForSortingMpi);
  ASSERT_EQ(ShellSortMpi->validation(), true);
  ShellSortMpi->pre_processing();
  ShellSortMpi->run();
  ShellSortMpi->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ShellSortMpi);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    EXPECT_TRUE(std::is_sorted(sorted_result_mpi.begin(), sorted_result_mpi.end()));
  }
}
