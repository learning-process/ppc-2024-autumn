#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shpynov_n_mismatched_characters_amount/include/mismatched_numbers_mpi.hpp"

TEST(mismatched_numbers_mpi, test_pipeline_run) {
  // create data
  boost::mpi::communicator world;

  std::string str1, str2 = "";
  std::vector<std::string> v1;
  std::vector<int> out(1, 0);

  std::string S = "qwerty", S1 = "qwertY";
  for (int i = 0; i < 100000; i++) {
    str1 += S;
    str2 += S1;
  }

  v1.push_back(str1);
  v1.push_back(str2);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(100000, out[0]);
  }
}

TEST(mismatched_numbers_mpi, test_task_run) {
  // create data
  boost::mpi::communicator world;

  std::string str1, str2 = "";
  std::vector<std::string> v1;
  std::vector<int> out(1, 0);

  std::string S = "qwerty", S1 = "qwertY";
  for (int i = 0; i < 100000; i++) {
    str1 += S;
    str2 += S1;
  }

  v1.push_back(str1);
  v1.push_back(str2);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
    taskDataPar->inputs_count.emplace_back(v1.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel = std::make_shared<shpynov_n_mismatched_numbers_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();
  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(100000, out[0]);
  }
}

TEST(mismatched_numbers_seq, test_pipeline_run) {
  // create data

  std::string str1, str2 = "";
  std::vector<std::string> v1;

  std::vector<int> out(1, 0);

  std::string S = "qwerty", S1 = "qwertY";

  for (int i = 0; i < 100000; i++) {
    str1 += S;
    str2 += S1;
  }
  v1.push_back(str1);
  v1.push_back(str2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<shpynov_n_mismatched_numbers_mpi::TestMPITaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
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

  ASSERT_EQ(100000, out[0]);
}

TEST(mismatched_numbers_seq, test_task_run) {
  // create data

  std::string str1, str2 = "";
  std::vector<std::string> v1;

  std::vector<int> out(1, 0);

  std::string S = "qwerty", S1 = "qwertY";

  for (int i = 0; i < 100000; i++) {
    str1 += S;
    str2 += S1;
  }
  v1.push_back(str1);
  v1.push_back(str2);

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[0].data()));
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(v1[1].data()));
  taskDataSeq->inputs_count.emplace_back(v1.size());
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  taskDataSeq->outputs_count.emplace_back(out.size());

  // Create Task
  auto testTaskSequential = std::make_shared<shpynov_n_mismatched_numbers_mpi::TestMPITaskSequential>(taskDataSeq);

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 1000;
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
  ASSERT_EQ(100000, out[0]);
}
