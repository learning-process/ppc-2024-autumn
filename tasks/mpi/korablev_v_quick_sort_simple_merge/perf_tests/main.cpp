#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <memory>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
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

TEST(korablev_v_quick_sort_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const size_t vector_size = 10000;
  auto random_vector = korablev_v_qucik_sort_simple_merge_mpi::generate_random_vector(vector_size);

  std::vector<size_t> in_size(1, vector_size);
  std::vector<int> out(vector_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_vector.data()));
    taskDataPar->inputs_count.emplace_back(random_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto parallelSort =
      std::make_shared<korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeParallel>(taskDataPar);
  ASSERT_EQ(parallelSort->validation(), true);
  parallelSort->pre_processing();
  parallelSort->run();
  parallelSort->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelSort);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(vector_size, out.size());
  }
}

TEST(korablev_v_quick_sort_mpi, test_task_run) {
  boost::mpi::communicator world;

  const size_t vector_size = 10000;
  auto random_vector = korablev_v_qucik_sort_simple_merge_mpi::generate_random_vector(vector_size);

  std::vector<size_t> in_size(1, vector_size);
  std::vector<int> out(vector_size, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in_size.data()));
    taskDataPar->inputs_count.emplace_back(in_size.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(random_vector.data()));
    taskDataPar->inputs_count.emplace_back(random_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto parallelSort =
      std::make_shared<korablev_v_qucik_sort_simple_merge_mpi::QuickSortSimpleMergeParallel>(taskDataPar);
  ASSERT_EQ(parallelSort->validation(), true);
  parallelSort->pre_processing();
  parallelSort->run();
  parallelSort->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 5;  // Запуск теста несколько раз для усреднения
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(parallelSort);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(vector_size, out.size());
  }
}