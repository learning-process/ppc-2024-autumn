#include "mpi/kudryashova_i_graham's_scan/include/Graham'sScanMPI.hpp"

#include <boost/mpi.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

namespace kudryashova_i_graham_scan_mpi {

bool kudryashova_i_graham_scan_mpi::TestMPITaskSequential::pre_processing() {
  internal_order_test();
  input_data.resize(taskData->inputs_count[0]);
  if (taskData->inputs[0] == nullptr || taskData->inputs_count[0] == 0) {
    return false;
  }
  auto* tmp_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
  std::copy(tmp_ptr, tmp_ptr + taskData->inputs_count[0], input_data.begin());
  return true;
}

bool kudryashova_i_graham_scan_mpi::TestMPITaskSequential::validation() {
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

std::vector<int8_t> kudryashova_i_graham_scan_mpi::TestMPITaskSequential::runGrahamScan(
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

bool kudryashova_i_graham_scan_mpi::TestMPITaskSequential::run() {
  internal_order_test();
  runGrahamScan(input_data);
  return true;
}

bool kudryashova_i_graham_scan_mpi::TestMPITaskSequential::post_processing() {
  internal_order_test();
  auto* outputData = reinterpret_cast<int8_t*>(taskData->outputs[0]);
  std::copy(result_vec.begin(), result_vec.end(), outputData);
  return true;
}

bool kudryashova_i_graham_scan_mpi::TestMPITaskParallel::pre_processing() {
  internal_order_test();
  int remainder = 0;
  processes = world.size();
  int counting_proc = world.size() - 1;
  if (world.rank() == 0) {
    if (processes == 1 || (int)(taskData->inputs_count[0]) < processes) {
      delta = (taskData->inputs_count[0]) / 2;
    } else {
      delta = (taskData->inputs_count[0] / 2) / counting_proc;
      remainder = (taskData->inputs_count[0] / 2) % counting_proc;
    }
  }
  segments.resize(processes);
  if (world.rank() == 0) {
    segments[0] = delta;
    for (int i = 1; i < processes; ++i) {
      segments[i] = delta + (i <= remainder ? 1 : 0);
    }
  }
  if (world.rank() == 0) {
    input_data.resize(taskData->inputs_count[0]);
    if (taskData->inputs[0] == nullptr || taskData->inputs_count[0] == 0) {
      return false;
    }
    auto* source_ptr = reinterpret_cast<uint8_t*>(taskData->inputs[0]);
    std::copy(source_ptr, source_ptr + taskData->inputs_count[0], input_data.begin());
    sortPoints(input_data);
  }
  return true;
}

bool kudryashova_i_graham_scan_mpi::TestMPITaskParallel::validation() {
  internal_order_test();
  if (world.rank() == 0) {
    return taskData->inputs_count[0] >= 6 && taskData->inputs_count[0] % 2 == 0;
  }
  return true;
}

std::vector<int8_t> rearrangeAndSort(const std::vector<int8_t>& input) {
  std::vector<int8_t> x_values;
  std::vector<int8_t> y_values;
  for (unsigned long i = 0; i < input.size(); i += 2) {
    if (i < input.size()) {
      x_values.push_back(input[i]);
    }
    if (i + 1 < input.size()) {
      y_values.push_back(input[i + 1]);
    }
  }
  std::vector<int8_t> result;
  result.reserve(x_values.size() + y_values.size());
  result.insert(result.end(), x_values.begin(), x_values.end());
  result.insert(result.end(), y_values.begin(), y_values.end());
  return result;
}

std::vector<int8_t> kudryashova_i_graham_scan_mpi::TestMPITaskParallel::runGrahamScan(
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

bool kudryashova_i_graham_scan_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  broadcast(world, segments.data(), processes, 0);
  if (world.rank() == 0) {
    size_t count = taskData->inputs_count[0];
    size_t halfSize = count / 2;
    firstHalf.resize(halfSize);
    secondHalf.resize(count - halfSize);
    std::copy(input_data.begin(), input_data.begin() + halfSize, firstHalf.begin());
    std::copy(input_data.begin() + halfSize, input_data.begin() + count, secondHalf.begin());
    int pointer = 0;
    for (int proc = 1; proc < world.size(); ++proc) {
      int proc_segment = segments[proc];
      world.send(proc, 0, firstHalf.data() + pointer, proc_segment);
      world.send(proc, 1, secondHalf.data() + pointer, proc_segment);
      pointer += proc_segment;
    }
  }
  if (world.rank() != 0) {
    local_input1_.resize(segments[world.rank()]);
    local_input2_.resize(segments[world.rank()]);
    world.recv(0, 0, local_input1_.data(), segments[world.rank()]);
    world.recv(0, 1, local_input2_.data(), segments[world.rank()]);
    std::vector<int8_t> local_GrahamScan_data;
    local_GrahamScan_data.reserve(local_input1_.size() + local_input2_.size());
    local_GrahamScan_data.insert(local_GrahamScan_data.end(), local_input1_.begin(), local_input1_.end());
    local_GrahamScan_data.insert(local_GrahamScan_data.end(), local_input2_.begin(), local_input2_.end());
    local_result = runGrahamScan(local_GrahamScan_data);
  }
  size_t local_size = local_result.size();
  std::vector<size_t> sizes(world.size());
  world.barrier();
  gather(world, local_size, sizes, 0);
  std::vector<int8_t> full_results;
  std::vector<size_t> displacements(world.size(), 0);
  if (world.rank() == 0) {
    size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
    full_results.resize(total_size);
    displacements[0] = 0;
    for (int i = 1; i < world.size(); ++i) {
      displacements[i] = displacements[i - 1] + sizes[i - 1];
    }
    if ((int)(taskData->inputs_count[0]) < world.size() || (world.size() == 1)) {
      runGrahamScan(input_data);
      return true;
    }
  } else {
    world.send(0, 0, local_result.data(), local_size);
  }
  if (world.rank() == 0) {
    for (int i = 1; i < world.size(); ++i) {
      world.recv(i, 0, full_results.data() + displacements[i], sizes[i]);
    }
    std::copy(local_result.begin(), local_result.end(), full_results.data() + displacements[world.rank()]);
    std::vector<int8_t> full_results_sort = rearrangeAndSort(full_results);
    runGrahamScan(full_results_sort);
  }
  return true;
}

bool kudryashova_i_graham_scan_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    if (!taskData->outputs.empty()) {
      auto* outputData = reinterpret_cast<int8_t*>(taskData->outputs[0]);
      std::copy(result_vec.begin(), result_vec.end(), outputData);
    } else {
      return false;
    }
  }
  return true;
}
}  // namespace kudryashova_i_graham_scan_mpi