#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/anufriev_d_star_topology/include/ops_mpi_anufriev.hpp"

std::vector<int> createInputVector(size_t size, int initialValue = 0, int step = 1) {
  std::vector<int> vec(size);
  std::iota(vec.begin(), vec.end(), initialValue);
  for (int& val : vec) {
    val *= step;
  }
  return vec;
}

std::vector<int> generate_random_vector(size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dist(-1000, 1000);
  std::vector<int> result(size);
  std::generate(result.begin(), result.end(), [&]() { return dist(gen); });
  return result;
}

#define PERF_TEST(test_name, data_size, num_runs, perf_method)                               \
    TEST(anufriev_d_star_topology_perf, test_name) {                                         \
    boost::mpi::communicator world;                                                          \
    std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>(); \
    std::vector<int> input_data;                                                             \
    std::vector<int> output_data;                                                            \
    if (world.rank() == 0) {                                                                 \
      input_data = createInputVector(data_size);                                             \
      taskData->inputs.push_back(reinterpret_cast<uint8_t*>(input_data.data()));             \
      taskData->inputs_count.push_back(input_data.size());                                   \
      output_data.resize(input_data.size());                                                 \
      taskData->outputs.push_back(reinterpret_cast<uint8_t*>(output_data.data()));           \
      taskData->outputs_count.push_back(output_data.size());                                 \
    }                                                                                        \
    auto task = std::make_shared<anufriev_d_star_topology::SimpleIntMPI>(taskData);          \
    auto perfAttr = std::make_shared<ppc::core::PerfAttr>();                                 \
    perfAttr->num_running = num_runs;                                                        \
    const boost::mpi::timer current_timer;                                                   \
    perfAttr->current_timer = [&] { return current_timer.elapsed(); };                       \
    auto perfResults = std::make_shared<ppc::core::PerfResults>();                           \
    auto perfAnalyzer = std::make_shared<ppc::core::Perf>(task);                             \
    perfAnalyzer->perf_method(perfAttr, perfResults);                                        \
    if (world.rank() == 0) {                                                                 \
      ppc::core::Perf::print_perf_statistic(perfResults);                                    \
    }                                                                                        \
  }

PERF_TEST(Pipeline_Small, 1000, 10, pipeline_run);
PERF_TEST(Pipeline_Medium, 100000, 5, pipeline_run);
PERF_TEST(Pipeline_Large, 10000000, 1, pipeline_run);
PERF_TEST(Pipeline_Random, 5000000, 3, pipeline_run);

PERF_TEST(Task_Small, 1000, 10, task_run);
PERF_TEST(Task_Medium, 100000, 5, task_run);
PERF_TEST(Task_Large, 10000000, 1, task_run);
PERF_TEST(Task_Random, 5000000, 3, task_run);

#undef PERF_TEST