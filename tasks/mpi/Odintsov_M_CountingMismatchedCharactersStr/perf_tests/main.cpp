#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/Odintsov_M_CountingMismatchedCharactersStr/include/ops_mpi.hpp"

static std::string get_random_str(size_t sz) {
  const char characters[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrswxyz0123456789";
  std::string str;

  std::srand(std::time(nullptr));

  for (size_t i = 0; i < sz; ++i) {
    int index = std::rand() % (sizeof(characters) - 1);
    str = characters[index];
  }

  return str;
}

TEST(Odintsov_m_MPI_parallel_perf_test, my_test_pipeline_run) {
  boost::mpi::communicator com;
  std::string s1 = get_random_str(2400);
  std::string s2 = get_random_str(2400);
  std::vector<char*> in{s1.data(), s2.data()};
  std::vector<size_t> out(1, 1);

  // Create Task Data Parallel
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto testClassPar =
      std::make_shared<Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel>(taskDataPar);
  ASSERT_EQ(testClassPar->validation(), true);
  testClassPar->pre_processing();
  testClassPar->run();
  testClassPar->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClassPar);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (com.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}

TEST(Odintsov_m_MPI_parallel_perf_test, my_test_task_run) {
  boost::mpi::communicator com;
  std::string s1 = get_random_str(2400);
  std::string s2 = get_random_str(2400);
  std::vector<char*> in{s1.data(), s2.data()};
  std::vector<size_t> out(1, 1);

  // Create Task Data Parallel//
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[0]));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in[1]));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }
  // Create Task
  auto testClassPar =
      std::make_shared<Odintsov_M_CountingMismatchedCharactersStr_mpi::CountingCharacterMPIParallel>(taskDataPar);
  ASSERT_EQ(testClassPar->validation(), true);
  testClassPar->pre_processing();
  testClassPar->run();
  testClassPar->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testClassPar);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (com.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}