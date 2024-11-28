#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/komshina_d_min_of_vector_elements/include/ops_mpi.hpp"

TEST(komshina_d_min_of_vector_elements_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int expected_min;
  if (world.rank() == 0) {
    const int count = 5000000;
    const int start_value = 1000000;
    const int decrement = 100;
    expected_min = start_value - decrement * (count - 1);
    global_vec.resize(count);
    for (int i = 0; i < count; ++i) {
      global_vec[i] = start_value - i * decrement;
    }
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto MinOfVectorElementTaskParallel =
      std::make_shared<komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel>(taskDataPar);
  ASSERT_EQ(MinOfVectorElementTaskParallel->validation(), true);
  MinOfVectorElementTaskParallel->pre_processing();
  MinOfVectorElementTaskParallel->run();
  MinOfVectorElementTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MinOfVectorElementTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected_min, global_min[0]);
  }
}

TEST(komshina_d_min_of_vector_elements_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int32_t> global_min(1, INT_MAX);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int expected_min;
  if (world.rank() == 0) {
    const int count = 5000000;
    const int start_value = 1000000;
    const int decrement = 100;
    expected_min = start_value - decrement * (count - 1);
    global_vec.resize(count);
    for (int i = 0; i < count; ++i) {
      global_vec[i] = start_value - i * decrement;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_min.data()));
    taskDataPar->outputs_count.emplace_back(global_min.size());
  }

  auto MinOfVectorElementTaskParallel =
      std::make_shared<komshina_d_min_of_vector_elements_mpi::MinOfVectorElementTaskParallel>(taskDataPar);
  ASSERT_EQ(MinOfVectorElementTaskParallel->validation(), true);
  MinOfVectorElementTaskParallel->pre_processing();
  MinOfVectorElementTaskParallel->run();
  MinOfVectorElementTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(MinOfVectorElementTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected_min, global_min[0]);
  }
}