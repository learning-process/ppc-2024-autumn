#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gnitienko_k_contrast_enhancement/include/ops_mpi.hpp"

TEST(gnitienko_k_contrast_enhancement_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  double contrast_factor = 2;
  std::vector<int> global_res;
  std::vector<int> expected(1000000, 182);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int img_size;
  if (world.rank() == 0) {
    img_size = 1000000;
    global_vec = std::vector<int>(img_size, 155);
    global_res = std::vector<int>(img_size, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected, global_res);
  }
}

TEST(gnitienko_k_contrast_enhancement_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  double contrast_factor = 2;
  std::vector<int> global_res;
  std::vector<int> expected(1000000, 182);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int img_size;
  if (world.rank() == 0) {
    img_size = 1000000;
    global_vec = std::vector<int>(img_size, 155);
    global_res = std::vector<int>(img_size, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(expected, global_res);
  }
}