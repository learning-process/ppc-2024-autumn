#include <gtest/gtest.h>

#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/leontev_n_image_enhancement/include/ops_seq.hpp"

template <class InOutType>
void taskEmplacement(std::shared_ptr<ppc::core::TaskData> &taskDataPar, std::vector<InOutType> &global_vec,
                     std::vector<InOutType> &global_sum) {
  taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
  taskDataPar->inputs_count.emplace_back(global_vec.size());
  taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_sum.data()));
  taskDataPar->outputs_count.emplace_back(global_sum.size());
}

TEST(leontev_n_image_enhancement_seq, test_pipeline_run) {
  const int width = 4000;
  const int height = 4000;
  const int count_size_vector = width * height * 3;

  std::vector<int> in_vec(count_size_vector, 0);
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto imgEnhancementSequential =
      std::make_shared<leontev_n_image_enhancement_seq::ImgEnhancementSequential>(taskDataSeq);
  taskEmplacement(taskDataSeq, in_vec, out_vec);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(imgEnhancementSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res_exp_out, out_vec);
}

TEST(leontev_n_image_enhancement_seq, test_task_run) {
  const int width = 4000;
  const int height = 4000;
  const int count_size_vector = width * height * 3;

  std::vector<int> in_vec(count_size_vector, 0);
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out(count_size_vector, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto imgEnhancementSequential =
      std::make_shared<leontev_n_image_enhancement_seq::ImgEnhancementSequential>(taskDataSeq);
  taskEmplacement(taskDataSeq, in_vec, out_vec);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(imgEnhancementSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res_exp_out, out_vec);
}
