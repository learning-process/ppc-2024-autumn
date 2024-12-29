// Filatev Vladislav Metod Belmana Forda
#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/filatev_v_metod_belmana_forda/include/ops_seq.hpp"

TEST(filatev_v_metod_belmana_forda_seq, test_pipeline_run) {
  int n = 10000;
  int m = n * (n - 1);
  int start = 0;
  std::vector<int> Adjncy(m, 0);
  std::vector<int> Xadj(n + 1);
  std::vector<int> Eweights(m, 1);
  std::vector<int> d(n);

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

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
  taskData->inputs_count.emplace_back(n);
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(start);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
  taskData->outputs_count.emplace_back(n);

  auto metodBelmanaForda = std::make_shared<filatev_v_metod_belmana_forda_seq::MetodBelmanaForda>(taskData);

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
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<int> tResh(n, 1);
  tResh[0] = 0;

  ASSERT_EQ(tResh, d);
}

TEST(filatev_v_metod_belmana_forda_seq, test_task_run) {
  int n = 10000;
  int m = n * (n - 1);
  int start = 0;
  std::vector<int> Adjncy(m, 0);
  std::vector<int> Xadj(n + 1);
  std::vector<int> Eweights(m, 1);
  std::vector<int> d(n);

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

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Adjncy.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Xadj.data()));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(Eweights.data()));
  taskData->inputs_count.emplace_back(n);
  taskData->inputs_count.emplace_back(m);
  taskData->inputs_count.emplace_back(start);
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(d.data()));
  taskData->outputs_count.emplace_back(n);

  auto metodBelmanaForda = std::make_shared<filatev_v_metod_belmana_forda_seq::MetodBelmanaForda>(taskData);

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
  ppc::core::Perf::print_perf_statistic(perfResults);

  std::vector<int> tResh(n, 1);
  tResh[0] = 0;

  ASSERT_EQ(tResh, d);
}
