#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/burykin_m_broadcast_mpi/include/ops_mpi.hpp"

namespace burykin_m_broadcast_mpi_mpi {

void fillVector(std::vector<int>& vector, int min_val, int max_val) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(min_val, max_val);
  for (size_t iter = 0; iter < vector.size();) {
    vector[iter++] = dist(gen);
  }
}

}  // namespace burykin_m_broadcast_mpi_mpi

TEST(burykin_m_broadcast_mpi_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int source_worker = 0;
  int data_size = 1000000;
  int min_val = -1000;
  int max_val = 1000;

  std::vector<int> recv_vorker(data_size);
  std::vector<int> result_vector(data_size);
  int global_max;

  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();

  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&source_worker));
  task->inputs_count.emplace_back(1);

  if (world.rank() == source_worker) {
    burykin_m_broadcast_mpi_mpi::fillVector(recv_vorker, min_val, max_val);

    task->inputs.emplace_back(reinterpret_cast<uint8_t*>(recv_vorker.data()));
    task->inputs_count.emplace_back(recv_vorker.size());
  }

  task->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vector.data()));
  task->outputs_count.emplace_back(result_vector.size());

  if (world.rank() == source_worker) {
    task->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_max));
    task->outputs_count.emplace_back(1);
  }

  auto taskForProcessing = std::make_shared<burykin_m_broadcast_mpi_mpi::StdBroadcastMPI>(task);
  bool val_res = taskForProcessing->validation();
  boost::mpi::broadcast(world, val_res, source_worker);
  if (val_res) {
    taskForProcessing->pre_processing();
    taskForProcessing->run();
    taskForProcessing->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };
    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskForProcessing);
    perfAnalyzer->pipeline_run(perfAttr, perfResults);
    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
    }

    boost::mpi::broadcast(world, recv_vorker, source_worker);

    EXPECT_EQ(recv_vorker, result_vector);
    if (world.size() == source_worker) {
      int result_max = *std::max_element(recv_vorker.begin(), recv_vorker.end());
      EXPECT_EQ(global_max, result_max);
    }
  }
}

TEST(burykin_m_broadcast_mpi_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  int source_worker = 0;
  int data_size = 1000000;
  int min_val = -1000;
  int max_val = 1000;

  std::vector<int> recv_vorker(data_size);
  std::vector<int> result_vector(data_size);
  int global_max;

  std::shared_ptr<ppc::core::TaskData> task = std::make_shared<ppc::core::TaskData>();

  task->inputs.emplace_back(reinterpret_cast<uint8_t*>(&source_worker));
  task->inputs_count.emplace_back(1);

  if (world.rank() == source_worker) {
    burykin_m_broadcast_mpi_mpi::fillVector(recv_vorker, min_val, max_val);

    task->inputs.emplace_back(reinterpret_cast<uint8_t*>(recv_vorker.data()));
    task->inputs_count.emplace_back(recv_vorker.size());
  }

  task->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vector.data()));
  task->outputs_count.emplace_back(result_vector.size());

  if (world.rank() == source_worker) {
    task->outputs.emplace_back(reinterpret_cast<uint8_t*>(&global_max));
    task->outputs_count.emplace_back(1);
  }

  auto taskForProcessing = std::make_shared<burykin_m_broadcast_mpi_mpi::StdBroadcastMPI>(task);
  bool val_res = taskForProcessing->validation();
  boost::mpi::broadcast(world, val_res, source_worker);
  if (val_res) {
    taskForProcessing->pre_processing();
    taskForProcessing->run();
    taskForProcessing->post_processing();

    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
    perfAttr->num_running = 10;
    const boost::mpi::timer current_timer;
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };
    auto perfResults = std::make_shared<ppc::core::PerfResults>();
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskForProcessing);
    perfAnalyzer->task_run(perfAttr, perfResults);
    if (world.rank() == 0) {
      ppc::core::Perf::print_perf_statistic(perfResults);
    }

    boost::mpi::broadcast(world, recv_vorker, source_worker);

    EXPECT_EQ(recv_vorker, result_vector);
    if (world.size() == source_worker) {
      int result_max = *std::max_element(recv_vorker.begin(), recv_vorker.end());
      EXPECT_EQ(global_max, result_max);
    }
  }
}
