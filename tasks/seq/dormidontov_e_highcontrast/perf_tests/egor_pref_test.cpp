#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/dormidontov_e_highcontrast/include/egor_include.hpp"

namespace dormidontov_e_highcontrast_seq {
std::vector<int> generate_pic(int heigh, int width) {
  std::vector<int> tmp(heigh * width);
  for (int i = 0; i < heigh; ++i) {
    for (int j = 0; j < width; ++j) {
      tmp[i * width + j] = (i * width + j) % 6;
    }
  }
  return tmp;
}
std::vector<int> generate_answer(int heigh, int width) {
  std::vector<int> tmp(heigh * width);
  for (int i = 0; i < heigh; ++i) {
    for (int j = 0; j < width; ++j) {
      tmp[i * width + j] = ((i * width + j) % 6) * 51;
    }
  }
  return tmp;
}
}  // namespace dormidontov_e_highcontrast_seq

TEST(dormidontov_e_highcontrast_seq, test_pipeline_run) {
  int height = 3000;
  int width = 3000;
  std::vector<int> picture = dormidontov_e_highcontrast_seq::generate_pic(height, width);
  std::vector<int> exp_res = dormidontov_e_highcontrast_seq::generate_answer(height, width);
  std::vector<int> res_out(height * width);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  // Create Task
  auto ContrastS = std::make_shared<dormidontov_e_highcontrast_seq::ContrastS>(taskDataSeq);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
  taskDataSeq->inputs_count.emplace_back(height * width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));
  taskDataSeq->outputs_count.emplace_back(res_out.size());

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ContrastS);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res_out == exp_res, true);
}

TEST(dormidontov_e_highcontrast_seq, test_task_run) {
  int height = 3000;
  int width = 3000;
  std::vector<int> picture = dormidontov_e_highcontrast_seq::generate_pic(height, width);
  std::vector<int> exp_res = dormidontov_e_highcontrast_seq::generate_answer(height, width);
  std::vector<int> res_out(height * width);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto ContrastS = std::make_shared<dormidontov_e_highcontrast_seq::ContrastS>(taskDataSeq);

  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
  taskDataSeq->inputs_count.emplace_back(height * width);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out.data()));
  taskDataSeq->outputs_count.emplace_back(res_out.size());

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ContrastS);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res_out == exp_res, true);
}