#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sidorina_p_broadcast/include/ops_mpi.hpp"

TEST(sidorina_p_broadcast_mpi, test_pipeline_run_m) {
  boost::mpi::communicator world;

  std::vector<int> reference;
  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int sz1 = 1000000;
    int sz2 = 1000;
    array.resize(sz1);
    reference.resize(sz1);
    terms.resize(sz2, 1);
    result.resize(sz1);
    for (int i = 0; i < sz1; i++) {
      array[i] = i;
      for (int j = 0; j < sz2; j++) {
        int a = array[i] + terms[j];
        reference[i] += a;
      }
    }

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskData->inputs_count.emplace_back(array.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskData->inputs_count.emplace_back(terms.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskData->outputs_count.emplace_back(result.size());
  }

  auto testMpiTaskParallel = std::make_shared<sidorina_p_broadcast_mpi::Broadcast>(taskData);
  testMpiTaskParallel->broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    sidorina_p_broadcast_mpi::Broadcast::broadcast_m(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(reference, result);
  }
}

TEST(sidorina_p_broadcast_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  std::vector<int> reference;
  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int sz1 = 1000000;
    int sz2 = 1000;
    array.resize(sz1);
    reference.resize(sz1);
    terms.resize(sz2, 1);
    result.resize(sz1);
    for (int i = 0; i < sz1; i++) {
      array[i] = i;
      for (int j = 0; j < sz2; j++) {
        int a = array[i] + terms[j];
        reference[i] += a;
      }
    }

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskData->inputs_count.emplace_back(array.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskData->inputs_count.emplace_back(terms.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskData->outputs_count.emplace_back(result.size());
  }

  auto testMpiTaskParallel = std::make_shared<sidorina_p_broadcast_mpi::Broadcast>(taskData);
  testMpiTaskParallel->broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    boost::mpi::broadcast(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(reference, result);
  }
}

TEST(sidorina_p_broadcast_mpi, test_task_run_m) {
  boost::mpi::communicator world;

  std::vector<int> reference;
  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int sz1 = 1000000;
    int sz2 = 1000;
    array.resize(sz1);
    reference.resize(sz1);
    terms.resize(sz2, 1);
    result.resize(sz1);
    for (int i = 0; i < sz1; i++) {
      array[i] = i;
      for (int j = 0; j < sz2; j++) {
        int a = array[i] + terms[j];
        reference[i] += a;
      }
    }

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskData->inputs_count.emplace_back(array.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskData->inputs_count.emplace_back(terms.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskData->outputs_count.emplace_back(result.size());
  }

  auto testMpiTaskParallel = std::make_shared<sidorina_p_broadcast_mpi::Broadcast>(taskData);
  testMpiTaskParallel->broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    sidorina_p_broadcast_mpi::Broadcast::broadcast_m(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(reference, result);
  }
}

TEST(sidorina_p_broadcast_mpi, test_task_run) {
  boost::mpi::communicator world;

  std::vector<int> reference;
  std::vector<int> array;
  std::vector<int> terms;
  std::vector<int> result;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    int sz1 = 1000000;
    int sz2 = 1000;
    array.resize(sz1);
    reference.resize(sz1);
    terms.resize(sz2, 1);
    result.resize(sz1);
    for (int i = 0; i < sz1; i++) {
      array[i] = i;
      for (int j = 0; j < sz2; j++) {
        int a = array[i] + terms[j];
        reference[i] += a;
      }
    }

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(array.data()));
    taskData->inputs_count.emplace_back(array.size());
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(terms.data()));
    taskData->inputs_count.emplace_back(terms.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(result.data()));
    taskData->outputs_count.emplace_back(result.size());
  }

  auto testMpiTaskParallel = std::make_shared<sidorina_p_broadcast_mpi::Broadcast>(taskData);
  testMpiTaskParallel->broadcast_fn = [](const boost::mpi::communicator& comm, int* values, int n, int root) {
    boost::mpi::broadcast(comm, values, n, root);
  };
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(reference, result);
  }
}