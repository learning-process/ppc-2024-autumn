#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/zaitsev_a_jarvis/include/ops_mpi.hpp"
#include "seq/zaitsev_a_jarvis/include/point.hpp"

namespace zaitsev_a_jarvis_mpi {
std::vector<zaitsev_a_jarvis_seq::Point<int>> get_random_vector(int size) {
  std::vector<zaitsev_a_jarvis_seq::Point<int>> vec(size, {0, 0});
  std::random_device dev;
  std::mt19937 gen(dev());
  for (int i = 0; i < size; i++) {
    vec[i].x = gen() % 100;
    vec[i].y = gen() % 100;
  }
  return vec;
}
}  // namespace zaitsev_a_jarvis_mpi

TEST(zaitsev_a_jarvis_mpi_perf, test_pipeline_run) {
  boost::mpi::communicator world;

  std::vector<zaitsev_a_jarvis_seq::Point<int>> in;
  std::vector<zaitsev_a_jarvis_seq::Point<int>> out(1, {0, 0});
  std::vector<zaitsev_a_jarvis_seq::Point<int>> expected;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  int size;
  if (world.rank() == 0) {
    size = 1e6;
    in = zaitsev_a_jarvis_mpi::get_random_vector(size);
    out.resize(size);

    in[0] = zaitsev_a_jarvis_seq::Point<int>{-1, -1};
    in[1] = zaitsev_a_jarvis_seq::Point<int>{-1, 101};
    in[2] = zaitsev_a_jarvis_seq::Point<int>{101, 101};
    in[3] = zaitsev_a_jarvis_seq::Point<int>{101, -1};

    expected = {{-1, -1}, {101, -1}, {101, 101}, {-1, 101}};

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto test = std::make_shared<zaitsev_a_jarvis_mpi::Jarvis<int>>(taskData);
  ASSERT_TRUE(test->validation());
  test->pre_processing();
  test->run();
  test->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    out.resize(taskData->outputs_count[0]);

    ASSERT_EQ(out, expected);
  }
}

TEST(zaitsev_a_jarvis_mpi_perf, test_task_run) {
  boost::mpi::communicator world;

  std::vector<zaitsev_a_jarvis_seq::Point<int>> in;
  std::vector<zaitsev_a_jarvis_seq::Point<int>> out(1, {0, 0});
  std::vector<zaitsev_a_jarvis_seq::Point<int>> expected;

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  int size;
  if (world.rank() == 0) {
    size = 1e6;
    in = zaitsev_a_jarvis_mpi::get_random_vector(size);
    out.resize(size);

    in[0] = zaitsev_a_jarvis_seq::Point<int>{-1, -1};
    in[1] = zaitsev_a_jarvis_seq::Point<int>{-1, 101};
    in[2] = zaitsev_a_jarvis_seq::Point<int>{101, 101};
    in[3] = zaitsev_a_jarvis_seq::Point<int>{101, -1};

    expected = {{-1, -1}, {101, -1}, {101, 101}, {-1, 101}};

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskData->inputs_count.emplace_back(in.size());

    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto test = std::make_shared<zaitsev_a_jarvis_mpi::Jarvis<int>>(taskData);
  ASSERT_EQ(test->validation(), true);
  test->pre_processing();
  test->run();
  test->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(test);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    out.resize(taskData->outputs_count[0]);

    ASSERT_EQ(out, expected);
  }
}