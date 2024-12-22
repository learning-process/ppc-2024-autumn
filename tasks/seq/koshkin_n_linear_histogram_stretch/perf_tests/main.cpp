#include <gtest/gtest.h>

#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/koshkin_n_linear_histogram_stretch/include/ops_seq.hpp"

TEST(koshkin_n_linear_histogram_stretch_seq, test_pipeline_run) {
  const int width = 4000;
  const int height = 4000;
  const int count_size_vector = width * height * 3;

  std::vector<int> in_vec = koshkin_n_linear_histogram_stretch_seq::getRandomImage(count_size_vector);
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out(count_size_vector, 0);

  int Imin = 255;
  int Imax = 0;
  std::vector<int> intensity(width * height);
  for (int i = 0, k = 0; i < count_size_vector; i += 3, ++k) {
    int R = in_vec[i];
    int G = in_vec[i + 1];
    int B = in_vec[i + 2];
    intensity[k] = static_cast<int>(0.299 * R + 0.587 * G + 0.114 * B);
    Imin = std::min(Imin, intensity[k]);
    Imax = std::max(Imax, intensity[k]);
  }

  for (int i = 0, k = 0; i < count_size_vector; i += 3, ++k) {
    if (Imin == Imax) {
      res_exp_out[i] = in_vec[i];
      res_exp_out[i + 1] = in_vec[i + 1];
      res_exp_out[i + 2] = in_vec[i + 2];
      continue;
    }
    int Inew = ((intensity[k] - Imin) * 255) / (Imax - Imin);
    float coeff = static_cast<float>(Inew) / static_cast<float>(intensity[k]);
    res_exp_out[i] = std::min(255, static_cast<int>(in_vec[i] * coeff));
    res_exp_out[i + 1] = std::min(255, static_cast<int>(in_vec[i + 1] * coeff));
    res_exp_out[i + 2] = std::min(255, static_cast<int>(in_vec[i + 2] * coeff));
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto testTaskSequential = std::make_shared<koshkin_n_linear_histogram_stretch_seq::TestTaskSequential>(taskDataSeq);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res_exp_out, out_vec);
}

TEST(koshkin_n_linear_histogram_stretch_seq, test_task_run) {
  const int width = 4000;
  const int height = 4000;
  const int count_size_vector = width * height * 3;

  std::vector<int> in_vec = koshkin_n_linear_histogram_stretch_seq::getRandomImage(count_size_vector);
  std::vector<int> out_vec(count_size_vector, 0);
  std::vector<int> res_exp_out(count_size_vector, 0);

  int Imin = 255;
  int Imax = 0;
  std::vector<int> intensity(width * height);
  for (int i = 0, k = 0; i < count_size_vector; i += 3, ++k) {
    int R = in_vec[i];
    int G = in_vec[i + 1];
    int B = in_vec[i + 2];
    intensity[k] = static_cast<int>(0.299 * R + 0.587 * G + 0.114 * B);
    Imin = std::min(Imin, intensity[k]);
    Imax = std::max(Imax, intensity[k]);
  }

  for (int i = 0, k = 0; i < count_size_vector; i += 3, ++k) {
    if (Imin == Imax) {
      res_exp_out[i] = in_vec[i];
      res_exp_out[i + 1] = in_vec[i + 1];
      res_exp_out[i + 2] = in_vec[i + 2];
      continue;
    }
    int Inew = ((intensity[k] - Imin) * 255) / (Imax - Imin);
    float coeff = static_cast<float>(Inew) / static_cast<float>(intensity[k]);
    res_exp_out[i] = std::min(255, static_cast<int>(in_vec[i] * coeff));
    res_exp_out[i + 1] = std::min(255, static_cast<int>(in_vec[i + 1] * coeff));
    res_exp_out[i + 2] = std::min(255, static_cast<int>(in_vec[i + 2] * coeff));
  }

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  auto testTaskSequential = std::make_shared<koshkin_n_linear_histogram_stretch_seq::TestTaskSequential>(taskDataSeq);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
  taskDataSeq->inputs_count.emplace_back(in_vec.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec.data()));
  taskDataSeq->outputs_count.emplace_back(out_vec.size());

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  ASSERT_EQ(res_exp_out, out_vec);
}
