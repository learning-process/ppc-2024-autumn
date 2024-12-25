#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "../include/ops_seq.hpp"
#include "core/perf/include/perf.hpp"

static pikarychev_i_gauss3_vertical_seq::Image CreateRandomImage(size_t width, size_t height) {
  auto img = pikarychev_i_gauss3_vertical_seq::Image::alloc(width, height);

  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<> distrib(0, 255);
  std::generate(img.data.begin(), img.data.end(), [&]() {
    return pikarychev_i_gauss3_vertical_seq::Image::Pixel{
        .r = (uint8_t)distrib(gen), .g = (uint8_t)distrib(gen), .b = (uint8_t)distrib(gen)};
  });

  return img;
}

TEST(pikarychev_i_gauss3_vertical_seq_perf_test, test_pipeline_run) {
  auto in = CreateRandomImage(391, 1221);
  auto out = pikarychev_i_gauss3_vertical_seq::Image::alloc(in.width, in.height);
  pikarychev_i_gauss3_vertical_seq::Kernel3x3 sharpness_kernel{0, -1, 0, -1, 5, -1, 0, -1, 0};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sharpness_kernel));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  auto task = std::make_shared<pikarychev_i_gauss3_vertical_seq::TaskSeq>(taskData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}

TEST(pikarychev_i_gauss3_vertical_seq_perf_test, test_task_run) {
  auto in = CreateRandomImage(391, 1300);
  auto out = pikarychev_i_gauss3_vertical_seq::Image::alloc(in.width, in.height);
  pikarychev_i_gauss3_vertical_seq::Kernel3x3 sharpness_kernel{0, -1, 0, -1, 5, -1, 0, -1, 0};

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&in));
  taskData->inputs.emplace_back(reinterpret_cast<uint8_t *>(&sharpness_kernel));
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out));

  auto task = std::make_shared<pikarychev_i_gauss3_vertical_seq::TaskSeq>(taskData);

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
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
}
