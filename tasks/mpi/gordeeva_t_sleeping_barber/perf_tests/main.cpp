#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gordeeva_t_sleeping_barber/include/ops_mpi.hpp"

TEST(gordeeva_t_sleeping_barber_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int max_waiting_chairs = 3;
  int global_res = -1;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  auto testMpiTaskParallel = std::make_shared<gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel>(taskDataPar);

  if (world.size() < 3) {
    ASSERT_EQ(testMpiTaskParallel->validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    ASSERT_TRUE(testMpiTaskParallel->pre_processing());

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = world.size() - 2;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    ASSERT_TRUE(testMpiTaskParallel->run());
    ASSERT_TRUE(testMpiTaskParallel->post_processing());

    world.barrier();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);

    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
      ASSERT_EQ(global_res, 0);
    }
  }
}

TEST(gordeeva_t_sleeping_barber_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int max_waiting_chairs = 3;
  int global_res = -1;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs_count.emplace_back(max_waiting_chairs);
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&global_res));
  taskDataPar->outputs_count.emplace_back(sizeof(global_res));

  auto testMpiTaskParallel = std::make_shared<gordeeva_t_sleeping_barber_mpi::TestMPITaskParallel>(taskDataPar);

  if (world.size() < 3) {
    ASSERT_EQ(testMpiTaskParallel->validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    ASSERT_TRUE(testMpiTaskParallel->pre_processing());

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = world.size() - 2;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };

    auto perfResults = std::make_shared<ppc::core::PerfResults>();

    ASSERT_TRUE(testMpiTaskParallel->run());
    ASSERT_TRUE(testMpiTaskParallel->post_processing());

    world.barrier();

    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
    perfAnalyzer->task_run(perfAttr, perfResults);

    if (world.rank() == 0) {
      ASSERT_EQ(global_res, 0);

      ppc::core::Perf::print_perf_statistic(perfResults);
    }
  }
}
