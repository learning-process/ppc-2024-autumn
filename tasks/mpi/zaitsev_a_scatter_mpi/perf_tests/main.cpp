#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zaitsev_a_scatter_handwritten/include/ops_mpi.hpp"

TEST(zaitsev_a_scatter_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> inp_vector;
  int extrema = -1;
  int root = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int sz;
  if (world.rank() == root) {
    sz = 10e6;
    inp_vector = std::vector<int>(sz, 1);
    inp_vector[0] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->inputs_count.emplace_back(inp_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->outputs_count.emplace_back(inp_vector.size());
  }

  auto task = std::make_shared<zaitsev_a_scatter::ScatterTask<int, MPI_Scatter>>(taskDataPar, root, MPI_INT);

  EXPECT_TRUE(task->validation());
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(extrema, reinterpret_cast<int*>(taskDataPar->outputs[0])[0]);
  }
}

TEST(zaitsev_a_scatter_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> inp_vector;
  int sz;
  int extrema = -1;
  int root = 0;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == root) {
    sz = 10e6;
    inp_vector = std::vector<int>(sz, 1);
    inp_vector[sz / 3] = extrema;
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->inputs_count.emplace_back(inp_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(inp_vector.data()));
    taskDataPar->outputs_count.emplace_back(inp_vector.size());
  }

  auto task = std::make_shared<zaitsev_a_scatter::ScatterTask<int, MPI_Scatter>>(taskDataPar, root, MPI_INT);

  ASSERT_TRUE(task->validation());
  task->pre_processing();
  task->run();
  task->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == root) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(extrema, reinterpret_cast<int*>(taskDataPar->outputs[0])[0]);
  }
}
