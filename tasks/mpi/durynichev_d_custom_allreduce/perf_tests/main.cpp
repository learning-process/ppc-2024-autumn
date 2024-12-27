#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/durynichev_d_custom_allreduce/include/ops_mpi.hpp"

namespace durynichev_d_custom_allreduce_mpi {

std::vector<int> genRundomVector(int n) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-1000, 1000);
  std::vector<int> vector(n);
  for (int i = 0; i < n; i++) {
    vector[i] = dist(gen);
  }
  return vector;
}

}  // namespace durynichev_d_custom_allreduce_mpi

TEST(durynichev_d_custom_allreduce_mpi, pipeline_run) {
  boost::mpi::communicator world;

  std::vector<int> data;
  int global_sum;
  int check;
  int n = 1000000;

  std::shared_ptr<ppc::core::TaskData> taskDataBroadcast = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data = durynichev_d_custom_allreduce_mpi::genRundomVector(n);
    check = std::accumulate(data.begin(), data.end(), 0);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);
  }

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_sum));
  taskDataBroadcast->outputs_count.emplace_back(1);

  auto taskBroadcast = std::make_shared<durynichev_d_custom_allreduce_mpi::MyAllreduceMPI>(taskDataBroadcast);
  ASSERT_TRUE(taskBroadcast->validation());
  taskBroadcast->pre_processing();
  taskBroadcast->run();
  taskBroadcast->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskBroadcast);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }

  boost::mpi::broadcast(world, check, 0);
  EXPECT_EQ(global_sum, check);
}

TEST(durynichev_d_custom_allreduce_mpi, task_run) {
  boost::mpi::communicator world;

  std::vector<int> data;
  int global_sum;
  int check;
  int n = 1000000;

  std::shared_ptr<ppc::core::TaskData> taskDataBroadcast = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    data = durynichev_d_custom_allreduce_mpi::genRundomVector(n);
    check = std::accumulate(data.begin(), data.end(), 0);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(data.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);
  }

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_sum));
  taskDataBroadcast->outputs_count.emplace_back(1);

  auto taskBroadcast = std::make_shared<durynichev_d_custom_allreduce_mpi::MyAllreduceMPI>(taskDataBroadcast);
  ASSERT_TRUE(taskBroadcast->validation());
  taskBroadcast->pre_processing();
  taskBroadcast->run();
  taskBroadcast->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskBroadcast);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }

  boost::mpi::broadcast(world, check, 0);
  EXPECT_EQ(global_sum, check);
}
