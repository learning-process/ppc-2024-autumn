// Filatev Vladislav Metod Belmana Forda
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/filatev_v_metod_belmana_forda/include/ops_mpi.hpp"

TEST(filatev_v_metod_belmana_forda_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int n = 9000;
  int m = n * (n - 1);
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy.resize(m, 0);
    Xadj.resize(n + 1);
    Eweights.resize(m, 1);
    d.resize(n);

    for (int i = 0, k = 0; i < n; i++) {
      Xadj[i] = k;
      for (int j = 0; j < n; j++) {
        if (i != j) {
          Adjncy[k] = j;
          k++;
        }
      }
    }
    Xadj[n] = m;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
    taskData->inputs_count.emplace_back(n);
    taskData->inputs_count.emplace_back(m);
    taskData->inputs_count.emplace_back(start);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
    taskData->outputs_count.emplace_back(n);
  }

  auto metodBelmanaForda = std::make_shared<filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI>(taskData);

  ASSERT_EQ(metodBelmanaForda->validation(), true);
  metodBelmanaForda->pre_processing();
  metodBelmanaForda->run();
  metodBelmanaForda->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(metodBelmanaForda);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> tResh(n, 1);
    tResh[0] = 0;

    ASSERT_EQ(tResh, d);
  }
}

TEST(filatev_v_metod_belmana_forda_mpi, test_task_run) {
  boost::mpi::communicator world;
  int n = 9000;
  int m = n * (n - 1);
  int start = 0;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    Adjncy.resize(m, 0);
    Xadj.resize(n + 1);
    Eweights.resize(m, 1);
    d.resize(n);

    for (int i = 0, k = 0; i < n; i++) {
      Xadj[i] = k;
      for (int j = 0; j < n; j++) {
        if (i != j) {
          Adjncy[k] = j;
          k++;
        }
      }
    }
    Xadj[n] = m;

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
    taskData->inputs_count.emplace_back(n);
    taskData->inputs_count.emplace_back(m);
    taskData->inputs_count.emplace_back(start);
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
    taskData->outputs_count.emplace_back(n);
  }

  auto metodBelmanaForda = std::make_shared<filatev_v_metod_belmana_forda_mpi::MetodBelmanaFordaMPI>(taskData);

  ASSERT_EQ(metodBelmanaForda->validation(), true);
  metodBelmanaForda->pre_processing();
  metodBelmanaForda->run();
  metodBelmanaForda->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(metodBelmanaForda);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> tResh(n, 1);
    tResh[0] = 0;

    ASSERT_EQ(tResh, d);
  }
}
