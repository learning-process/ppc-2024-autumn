#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/kazunin_n_dining_philosophers/include/ops_mpi.hpp"

namespace kazunin_n_dining_philosophers_mpi {

bool is_valid(int eat_limit, int min_think_time, int max_think_time, int min_eat_time, int max_eat_time) {
  return eat_limit > 0 && min_think_time < max_think_time && min_eat_time < max_eat_time && max_think_time < 100 &&
         min_think_time > 0 && max_eat_time < 100 && min_eat_time > 0;
}

}  // namespace kazunin_n_dining_philosophers_mpi

TEST(kazunin_n_dining_philosophers_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  int eat_limit = 3;
  int min_think_time = 20;
  int max_think_time = 40;
  int min_eat_time = 10;
  int max_eat_time = 20;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eat_limit));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&min_think_time));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_think_time));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&min_eat_time));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_eat_time));
  taskDataPar->inputs_count.emplace_back(1);

  auto taskParallel = std::make_shared<kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI>(taskDataPar);
  if (kazunin_n_dining_philosophers_mpi::is_valid(eat_limit, min_think_time, max_think_time, min_eat_time,
                                                  max_eat_time)) {
    EXPECT_TRUE(taskParallel->validation());
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
  } else {
    EXPECT_FALSE(taskParallel->validation());
  }
}

TEST(kazunin_n_dining_philosophers_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  if (world.size() < 4) {
    GTEST_SKIP();
  }

  int eat_limit = 3;
  int min_think_time = 20;
  int max_think_time = 40;
  int min_eat_time = 10;
  int max_eat_time = 20;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&eat_limit));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&min_think_time));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_think_time));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&min_eat_time));
  taskDataPar->inputs_count.emplace_back(1);

  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&max_eat_time));
  taskDataPar->inputs_count.emplace_back(1);

  auto taskParallel = std::make_shared<kazunin_n_dining_philosophers_mpi::DiningPhilosophersParallelMPI>(taskDataPar);
  if (kazunin_n_dining_philosophers_mpi::is_valid(eat_limit, min_think_time, max_think_time, min_eat_time,
                                                  max_eat_time)) {
    EXPECT_TRUE(taskParallel->validation());
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
  } else {
    EXPECT_FALSE(taskParallel->validation());
  }
}
