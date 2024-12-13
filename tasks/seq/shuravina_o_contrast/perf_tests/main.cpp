#include <gtest/gtest.h>

#include <chrono>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "seq/shuravina_o_contrast/include/ops_seq.hpp"

namespace shuravina_o_contrast {

std::vector<uint8_t> generateRandomImage(size_t size) {
  std::vector<uint8_t> image(size);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 255);
  for (size_t i = 0; i < size; ++i) {
    image[i] = dis(gen);
  }
  return image;
}

void runPerformanceTest(size_t count, const std::string& testName) {
  std::vector<uint8_t> in = generateRandomImage(count);
  std::vector<uint8_t> out(count, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
  taskDataSeq->inputs_count.emplace_back(in.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  auto contrastTaskSequential = std::make_shared<shuravina_o_contrast::ContrastTaskSequential>(taskDataSeq);
  ASSERT_EQ(contrastTaskSequential->validation(), true);
  contrastTaskSequential->pre_processing();
  contrastTaskSequential->run();
  contrastTaskSequential->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(contrastTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  std::cout << "Performance results for " << testName << ":" << std::endl;
  ppc::core::Perf::print_perf_statistic(perfResults);
}

}  // namespace shuravina_o_contrast

TEST(shuravina_o_contrast_perf, Test_Contrast_Enhancement_Small_Image) {
  shuravina_o_contrast::runPerformanceTest(1000, "Small Image");
}

TEST(shuravina_o_contrast_perf, Test_Contrast_Enhancement_Large_Image) {
  shuravina_o_contrast::runPerformanceTest(1000000, "Large Image");
}