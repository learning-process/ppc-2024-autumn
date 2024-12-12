// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vladimirova_j_gather/include/ops_mpi.hpp"
#include "mpi/vladimirova_j_gather/include/ops_mpi_not_my_gather.hpp"
using namespace std::chrono_literals;

namespace vladimirova_j_gather_mpi {

std::vector<int> getRandomVal(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<int> vec(sz);
  vec[0] = 2;
  vec[sz - 1] = 2;
  for (int i = 1; i < sz - 1; i++) {
    if ((i != 0) && (vec[i - 1] != 2)) {
      vec[i] = 2;
      continue;
    }
    vec[i] = (gen() % 3 - 1);
    if (vec[i] == 0) vec[i] = 2;
  }
  return vec;
};
}  // namespace vladimirova_j_gather_mpi

TEST(vladimirova_j_gather_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int d_end_count = 500;
  int noDEnd = 0;
  if (world.rank() == 0) {
    for (int j = 0; j < d_end_count; j++) {
      std::vector<int> some_dead_end;
      std::vector<int> tmp;
      some_dead_end = vladimirova_j_gather_mpi::getRandomVal(15);
      tmp = vladimirova_j_gather_mpi::getRandomVal(15);
      noDEnd += 15;
      global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
      global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
      global_vector.push_back(-1);
      global_vector.push_back(-1);
      noDEnd += 2;
      for (int i = some_dead_end.size() - 1; i >= 0; i--) {
        if (some_dead_end[i] != 2) {
          global_vector.push_back(-1 * some_dead_end[i]);
        } else {
          global_vector.push_back(2);
        }
      }
    }
  }

  std::vector<int32_t> ans_buf_vec(noDEnd);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<vladimirova_j_gather_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ((noDEnd >= (int)taskDataPar->outputs_count[0]), 1);
  }
}

TEST(vladimirova_j_gather_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int d_end_count = 900;
  int noDEnd = 0;
  if (world.rank() == 0) {
    for (int j = 0; j < d_end_count; j++) {
      std::vector<int> some_dead_end;
      std::vector<int> tmp;
      some_dead_end = vladimirova_j_gather_mpi::getRandomVal(15);
      tmp = vladimirova_j_gather_mpi::getRandomVal(15);
      noDEnd += 15;
      global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
      global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
      global_vector.push_back(-1);
      global_vector.push_back(-1);
      noDEnd += 2;
      for (int i = some_dead_end.size() - 1; i >= 0; i--) {
        if (some_dead_end[i] != 2) {
          global_vector.push_back(-1 * some_dead_end[i]);
        } else {
          global_vector.push_back(2);
        }
      }
    }
  }

  std::vector<int32_t> ans_buf_vec(noDEnd);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<vladimirova_j_gather_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ((noDEnd >= (int)taskDataPar->outputs_count[0]), 1);
  }
}

TEST(vladimirova_j_not_my_gather_mpi, test_pipeline_run_not_my_gather) {
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int d_end_count = 500;
  int noDEnd = 0;
  if (world.rank() == 0) {
    for (int j = 0; j < d_end_count; j++) {
      std::vector<int> some_dead_end;
      std::vector<int> tmp;
      some_dead_end = vladimirova_j_gather_mpi::getRandomVal(15);
      tmp = vladimirova_j_gather_mpi::getRandomVal(15);
      noDEnd += 15;
      global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
      global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
      global_vector.push_back(-1);
      global_vector.push_back(-1);
      noDEnd += 2;
      for (int i = some_dead_end.size() - 1; i >= 0; i--) {
        if (some_dead_end[i] != 2) {
          global_vector.push_back(-1 * some_dead_end[i]);
        } else {
          global_vector.push_back(2);
        }
      }
    }
  }

  std::vector<int32_t> ans_buf_vec(noDEnd);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<vladimirova_j_not_my_gather_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ((noDEnd >= (int)taskDataPar->outputs_count[0]), 1);
  }
}

TEST(vladimirova_j_not_my_gather_mpi, test_task_run_not_my_gather) {
  boost::mpi::communicator world;
  std::vector<int> global_vector;
  std::vector<int32_t> global_sum(1, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int d_end_count = 900;
  int noDEnd = 0;
  if (world.rank() == 0) {
    for (int j = 0; j < d_end_count; j++) {
      std::vector<int> some_dead_end;
      std::vector<int> tmp;
      some_dead_end = vladimirova_j_gather_mpi::getRandomVal(15);
      tmp = vladimirova_j_gather_mpi::getRandomVal(15);
      noDEnd += 15;
      global_vector.insert(global_vector.end(), tmp.begin(), tmp.end());
      global_vector.insert(global_vector.end(), some_dead_end.begin(), some_dead_end.end());
      global_vector.push_back(-1);
      global_vector.push_back(-1);
      noDEnd += 2;
      for (int i = some_dead_end.size() - 1; i >= 0; i--) {
        if (some_dead_end[i] != 2) {
          global_vector.push_back(-1 * some_dead_end[i]);
        } else {
          global_vector.push_back(2);
        }
      }
    }
  }

  std::vector<int32_t> ans_buf_vec(noDEnd);
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(global_vector.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(ans_buf_vec.data()));
    taskDataPar->outputs_count.emplace_back(1);
  }

  auto testMpiTaskParallel = std::make_shared<vladimirova_j_not_my_gather_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ((noDEnd >= (int)taskDataPar->outputs_count[0]), 1);
  }
}