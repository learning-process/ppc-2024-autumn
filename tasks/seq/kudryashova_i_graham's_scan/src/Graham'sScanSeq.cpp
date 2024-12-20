#include "seq/kudryashova_i_graham's_scan/include/Graham'sScanSeq.hpp"

namespace kudryashova_i_graham_scan_seq {

bool kudryashova_i_graham_scan_seq::TestTaskSequential::pre_processing() {
  internal_order_test();
  input_data.resize(taskData->inputs_count[0]);
  if (taskData->inputs[0] == nullptr || taskData->inputs_count[0] == 0) {
    return false;
  }
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_data.begin());
  return true;
}

bool kudryashova_i_graham_scan_seq::TestTaskSequential::validation() {
  internal_order_test();
  return taskData->inputs_count[0] >= 6 && taskData->inputs_count[0] % 2 == 0;
}

bool isCounterClockwise(const std::pair<int8_t, int8_t>& p1, const std::pair<int8_t, int8_t>& p2,
                        const std::pair<int8_t, int8_t>& p3) {
  return (p2.first - p1.first) * (p3.second - p1.second) > (p2.second - p1.second) * (p3.first - p1.first);
}

void sortPoints(std::vector<int8_t>& points) {
  int n = points.size() / 2;
  std::pair<int8_t, int8_t> p0(points[0], points[n]);
  std::vector<std::pair<int8_t, int8_t>> pointList;
  pointList.reserve(points.size());
  for (int i = 0; i < n; ++i) {
    pointList.emplace_back(points[i], points[n + i]);
  }
  std::sort(pointList.begin(), pointList.end(),
            [&p0](const std::pair<int8_t, int8_t>& a, const std::pair<int8_t, int8_t>& b) {
              if (atan2(a.second - p0.second, a.first - p0.first) == atan2(b.second - p0.second, b.first - p0.first)) {
                return ((a.first - p0.first) * (a.first - p0.first) + (a.second - p0.second) * (a.second - p0.second)) <
                       ((b.first - p0.first) * (b.first - p0.first) + (b.second - p0.second) * (b.second - p0.second));
              }
              return atan2(a.second - p0.second, a.first - p0.first) < atan2(b.second - p0.second, b.first - p0.first);
            });
  for (int i = 0; i < n; ++i) {
    points[i] = pointList[i].first;
    points[n + i] = pointList[i].second;
  }
}

std::vector<int8_t> kudryashova_i_graham_scan_seq::TestTaskSequential::runGrahamScan(
    std::vector<int8_t>& Graham_input_data) {
  std::vector<int8_t> hull;
  std::vector<int8_t> points = Graham_input_data;
  int n = points.size() / 2;
  int min_y_index = 0;
  for (int i = 1; i < n; ++i) {
    if (points[n + i] < points[n + min_y_index] ||
        (points[n + i] == points[n + min_y_index] && points[i] < points[min_y_index])) {
      min_y_index = i;
    }
  }
  std::swap(points[0], points[min_y_index]);
  std::swap(points[n], points[min_y_index + n]);
  std::vector<int> indices(n);
  for (int i = 1; i < n; ++i) indices[i] = i;
  sortPoints(points);
  hull.push_back(points[0]);
  hull.push_back(points[n]);
  for (int i = 1; i < n; ++i) {
    int index = indices[i];
    while (hull.size() >= 4 &&
           !isCounterClockwise({hull[hull.size() - 4], hull[hull.size() - 3]},
                               {hull[hull.size() - 2], hull[hull.size() - 1]}, {points[index], points[n + index]})) {
      hull.pop_back();
      hull.pop_back();
    }
    if (!isCounterClockwise({hull[hull.size() - 2], hull[hull.size() - 1]}, {points[index], points[n + index]},
                            {points[index], points[n + index]})) {
      hull.push_back(points[index]);
      hull.push_back(points[n + index]);
    }
  }
  result_vec.resize(hull.size());
  std::copy(hull.begin(), hull.end(), result_vec.data());
  return result_vec;
}

bool kudryashova_i_graham_scan_seq::TestTaskSequential::run() {
  internal_order_test();
  runGrahamScan(input_data);
  return true;
}

bool kudryashova_i_graham_scan_seq::TestTaskSequential::post_processing() {
  internal_order_test();
  auto* outputData = reinterpret_cast<int8_t*>(taskData->outputs[0]);
  std::copy(result_vec.begin(), result_vec.end(), outputData);
  return true;
}
}  // namespace kudryashova_i_graham_scan_seq