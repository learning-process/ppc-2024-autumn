#include <gtest/gtest.h>

#include "core/perf/include/perf.hpp"
#include "seq/kudryashova_i_graham's_scan/include/Graham'sScanSeq.hpp"

void generateUniquePoints(int numPoints, int8_t Ans_number, std::vector<int8_t> &xCoords,
                          std::vector<int8_t> &yCoords) {
  int8_t x_min = -(Ans_number - 1);
  int8_t x_max = (Ans_number - 1);
  int8_t y_min = -(Ans_number - 1);
  int8_t y_max = (Ans_number - 1);
  if (numPoints > (x_max - x_min + 1) * (y_max - y_min + 1)) {
    std::cerr << "Error: Not enough unique points can be generated in the given range." << std::endl;
    return;
  }
  std::vector<std::pair<int8_t, int8_t>> allPoints;
  for (int8_t x = x_min; x <= x_max; x += 1) {
    for (int8_t y = y_min; y <= y_max; y += 1) {
      allPoints.emplace_back(x, y);
    }
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::shuffle(allPoints.begin(), allPoints.end(), gen);
  for (int i = 0; i < numPoints; ++i) {
    xCoords.push_back(allPoints[i].first);
    yCoords.push_back(allPoints[i].second);
  }
  xCoords.push_back(-Ans_number);
  yCoords.push_back(-Ans_number);
  xCoords.push_back(Ans_number);
  yCoords.push_back(-Ans_number);
  xCoords.push_back(Ans_number);
  yCoords.push_back(Ans_number);
  xCoords.push_back(-Ans_number);
  yCoords.push_back(Ans_number);
}

TEST(kudryashova_i_graham_scan_seq, test_pipeline_run) {
  const int count_size = 60000;
  const int ans_number = 125;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> vector_x;
  std::vector<int8_t> vector_y;
  generateUniquePoints(count_size, ans_number, vector_x, vector_y);
  std::vector<int8_t> result(8, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  auto testTaskSequential = std::make_shared<kudryashova_i_graham_scan_seq::TestTaskSequential>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  std::vector<int8_t> answer = {-ans_number, -ans_number, ans_number,  -ans_number,
                                ans_number,  ans_number,  -ans_number, ans_number};
  ASSERT_EQ(answer, result);
}

TEST(kudryashova_i_graham_scan_seq, test_task_run) {
  const int count_size = 60000;
  const int ans_number = 125;
  std::vector<int8_t> global_vector;
  std::vector<int8_t> vector_x;
  std::vector<int8_t> vector_y;
  generateUniquePoints(count_size, ans_number, vector_x, vector_y);
  std::vector<int8_t> result(8, 0);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  global_vector.reserve(vector_x.size() + vector_y.size());
  global_vector.insert(global_vector.end(), vector_x.begin(), vector_x.end());
  global_vector.insert(global_vector.end(), vector_y.begin(), vector_y.end());
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  taskDataSeq->inputs_count.emplace_back(global_vector.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  taskDataSeq->outputs_count.emplace_back(result.size());
  auto testTaskSequential = std::make_shared<kudryashova_i_graham_scan_seq::TestTaskSequential>(taskDataSeq);
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perfAttr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testTaskSequential);
  perfAnalyzer->task_run(perfAttr, perfResults);
  ppc::core::Perf::print_perf_statistic(perfResults);
  std::vector<int8_t> answer = {-ans_number, -ans_number, ans_number,  -ans_number,
                                ans_number,  ans_number,  -ans_number, ans_number};
  ASSERT_EQ(answer, result);
}