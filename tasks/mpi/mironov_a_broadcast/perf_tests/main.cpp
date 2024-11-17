#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/mironov_a_broadcast/include/ops_mpi.hpp"

TEST(mironov_a_broadcast_mpi, boost_test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  
  if (world.rank() == 0) {
    int size1 = 1000000;
    int size2 = 100;
    global_input.resize(size1);
    golds.resize(size1);
    global_powers.resize(size2, 1);
    global_res.resize(size1);

    for (int i = 0; i < size1; i++) {
      global_input[i] = i;
      golds[i] = i * size2;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataPar->inputs_count.emplace_back(global_input.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataPar->inputs_count.emplace_back(global_powers.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<mironov_a_broadcast_mpi::ComponentSumPowerBoostImpl>(taskDataPar);
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
    ASSERT_EQ(golds, global_res);
  }
}

TEST(mironov_a_broadcast_mpi, boost_test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    int size1 = 1000000;
    int size2 = 100;
    global_input.resize(size1);
    golds.resize(size1);
    global_powers.resize(size2, 1);
    global_res.resize(size1);

    for (int i = 0; i < size1; i++) {
      global_input[i] = i;
      golds[i] = i * size2;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataPar->inputs_count.emplace_back(global_input.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataPar->inputs_count.emplace_back(global_powers.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<mironov_a_broadcast_mpi::ComponentSumPowerBoostImpl>(taskDataPar);
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
    ASSERT_EQ(golds, global_res);
  }
}
TEST(mironov_a_broadcast_mpi, custom_test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int size1 = 1000000;
    int size2 = 100;
    global_input.resize(size1);
    golds.resize(size1);
    global_powers.resize(size2, 1);
    global_res.resize(size1);

    for (int i = 0; i < size1; i++) {
      global_input[i] = i;
      golds[i] = i * size2;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataPar->inputs_count.emplace_back(global_input.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataPar->inputs_count.emplace_back(global_powers.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<mironov_a_broadcast_mpi::ComponentSumPowerCustomImpl>(taskDataPar);
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
    ASSERT_EQ(golds, global_res);
  }
}

TEST(mironov_a_broadcast_mpi, custom_test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> golds;
  std::vector<int> global_input;
  std::vector<int> global_powers;
  std::vector<int> global_res;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    int size1 = 1000000;
    int size2 = 100;
    global_input.resize(size1);
    golds.resize(size1);
    global_powers.resize(size2, 1);
    global_res.resize(size1);

    for (int i = 0; i < size1; i++) {
      global_input[i] = i;
      golds[i] = i * size2;
    }

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskDataPar->inputs_count.emplace_back(global_input.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_powers.data()));
    taskDataPar->inputs_count.emplace_back(global_powers.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<mironov_a_broadcast_mpi::ComponentSumPowerCustomImpl>(taskDataPar);
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
    ASSERT_EQ(golds, global_res);
  }
}
