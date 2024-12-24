#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/tyurin_m_linear_topology/include/ops_mpi.hpp"

TEST(tyurin_m_linear_topology_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int sender = 0;
  int target = 1;

  if (world.size() <= std::max(sender, target)) {
    int data = 4096;
    bool result_flag = false;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sender));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
    taskDataPar->inputs_count.emplace_back(1);

    if (world.rank() == sender) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data));
      taskDataPar->inputs_count.emplace_back(1);
    } else if (world.rank() == target) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_flag));
      taskDataPar->outputs_count.emplace_back(1);
    }

    auto taskParallel = std::make_shared<tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI>(taskDataPar);
    ASSERT_TRUE(taskParallel->validation());
    taskParallel->pre_processing();
    taskParallel->run();
    taskParallel->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);

    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
    }
    if (world.rank() == target) {
      EXPECT_TRUE(result_flag);
    }
  }
}

TEST(tyurin_m_linear_topology_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int sender = 0;
  int target = 1;

  if (world.size() <= std::max(sender, target)) {
    int data = 4096;
    bool result_flag = false;

    std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&sender));
    taskDataPar->inputs_count.emplace_back(1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&target));
    taskDataPar->inputs_count.emplace_back(1);

    if (world.rank() == sender) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&data));
      taskDataPar->inputs_count.emplace_back(1);
    } else if (world.rank() == target) {
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result_flag));
      taskDataPar->outputs_count.emplace_back(1);
    }

    auto taskParallel = std::make_shared<tyurin_m_linear_topology_mpi::LinearTopologyParallelMPI>(taskDataPar);
    ASSERT_TRUE(taskParallel->validation());
    taskParallel->pre_processing();
    taskParallel->run();
    taskParallel->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
    perfAnalyzer->task_run(perfAttr, perfResults);

    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
    }
    if (world.rank() == target) {
      EXPECT_TRUE(result_flag);
    }
  }
}