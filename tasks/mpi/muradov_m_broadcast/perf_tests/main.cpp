#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/muradov_m_broadcast/include/ops_mpi.hpp"

namespace muradov_m_broadcast_mpi {

std::vector<int> gen_rundom_vector(int n) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(-1000, 1000);
  std::vector<int> vector(n);
  for (int i = 0; i < n; i++) {
    vector[i] = dist(gen);
  }
  return vector;
}

}  // namespace muradov_m_broadcast_mpi

TEST(muradov_m_broadcast_mpi, pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int source = 0;

  int n = 100000;
  std::vector<int> A;
  std::vector<int> A_res(n);
  std::vector<int> B_res(n);
  int my_global_sum_A;
  int mpi_global_sum_B;

  std::shared_ptr<ppc::core::TaskData> taskDataBroadcast = std::make_shared<ppc::core::TaskData>();

  taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(&source));
  taskDataBroadcast->inputs_count.emplace_back(1);

  if (world.rank() == source) {
    A = muradov_m_broadcast_mpi::gen_rundom_vector(n);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);
  }

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(A_res.data()));
  taskDataBroadcast->outputs_count.emplace_back(n);

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(B_res.data()));
  taskDataBroadcast->outputs_count.emplace_back(n);

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(&my_global_sum_A));
  taskDataBroadcast->outputs_count.emplace_back(1);

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(&mpi_global_sum_B));
  taskDataBroadcast->outputs_count.emplace_back(1);

  auto taskBroadcast = std::make_shared<muradov_m_broadcast_mpi::BroadcastParallelMPI>(taskDataBroadcast);
  bool val_res = taskBroadcast->validation();
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

  if (val_res) {
    ASSERT_EQ(A_res, B_res);
    EXPECT_EQ(my_global_sum_A, mpi_global_sum_B);
  }
}

TEST(muradov_m_broadcast_mpi, task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  int source = 0;

  int n = 100000;
  std::vector<int> A;
  std::vector<int> A_res(n);
  std::vector<int> B_res(n);
  int my_global_sum_A;
  int mpi_global_sum_B;

  std::shared_ptr<ppc::core::TaskData> taskDataBroadcast = std::make_shared<ppc::core::TaskData>();

  taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(&source));
  taskDataBroadcast->inputs_count.emplace_back(1);

  if (world.rank() == source) {
    A = muradov_m_broadcast_mpi::gen_rundom_vector(n);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);

    taskDataBroadcast->inputs.emplace_back(reinterpret_cast<uint8_t*>(A.data()));
    taskDataBroadcast->inputs_count.emplace_back(n);
  }

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(A_res.data()));
  taskDataBroadcast->outputs_count.emplace_back(n);

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(B_res.data()));
  taskDataBroadcast->outputs_count.emplace_back(n);

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(&my_global_sum_A));
  taskDataBroadcast->outputs_count.emplace_back(1);

  taskDataBroadcast->outputs.emplace_back(reinterpret_cast<uint8_t*>(&mpi_global_sum_B));
  taskDataBroadcast->outputs_count.emplace_back(1);

  auto taskBroadcast = std::make_shared<muradov_m_broadcast_mpi::BroadcastParallelMPI>(taskDataBroadcast);
  bool val_res = taskBroadcast->validation();
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

  if (val_res) {
    ASSERT_EQ(A_res, B_res);
    EXPECT_EQ(my_global_sum_A, mpi_global_sum_B);
  }
}
