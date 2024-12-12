#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/beskhmelnova_k_jarvis_march/include/jarvis_march.hpp"

TEST(mpi_beskhmelnova_k_jarvis_march_perf_test, test_pipeline_run) {
  boost::mpi::communicator world;

  int num_points = 1000;
  std::vector<double> x;
  std::vector<double> y;

  int hull_size;
  std::vector<double> hull_x;
  std::vector<double> hull_y;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    x = std::vector<double>(num_points);
    y = std::vector<double>(num_points);
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 4; i < num_points; i++) {
      x[i] = std::rand() % 1000;
      y[i] = std::rand() % 1000;
    }
    x[0] = -1.0;
    y[0] = -1.0;

    x[1] = -1.0;
    y[1] = 1000.0;

    x[2] = 1000.0;
    y[2] = 1000.0;

    x[3] = 1000.0;
    y[3] = -1.0;

    hull_x = std::vector<double>(num_points);
    hull_y = std::vector<double>(num_points);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->inputs_count.emplace_back(x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataPar->inputs_count.emplace_back(y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
    taskDataPar->outputs_count.emplace_back(hull_x.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
    taskDataPar->outputs_count.emplace_back(hull_y.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<double>>(taskDataPar);
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
    int res_size = 4;
    std::vector<double> res_x = {-1.0, 1000.0, 1000.0, -1.0};
    std::vector<double> res_y = {-1.0, -1.0, 1000.0, 1000.0};
    for (int i = 0; i < res_size; i++) {
      EXPECT_EQ(res_x[i], hull_x[i]);
      EXPECT_EQ(res_y[i], hull_y[i]);
    }
  }
}

TEST(mpi_beskhmelnova_k_jarvis_march_perf_test, test_task_run) {
  boost::mpi::communicator world;

  int num_points = 1000;
  std::vector<double> x;
  std::vector<double> y;

  int hull_size;
  std::vector<double> hull_x;
  std::vector<double> hull_y;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    x = std::vector<double>(num_points);
    y = std::vector<double>(num_points);
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 4; i < num_points; i++) {
      x[i] = std::rand() % 1000;
      y[i] = std::rand() % 1000;
    }
    x[0] = -1.0;
    y[0] = -1.0;

    x[1] = -1.0;
    y[1] = 1000.0;

    x[2] = 1000.0;
    y[2] = 1000.0;

    x[3] = 1000.0;
    y[3] = -1.0;

    hull_x = std::vector<double>(num_points);
    hull_y = std::vector<double>(num_points);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(x.data()));
    taskDataPar->inputs_count.emplace_back(x.size());
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(y.data()));
    taskDataPar->inputs_count.emplace_back(y.size());

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&hull_size));
    taskDataPar->outputs_count.emplace_back((size_t)1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_x.data()));
    taskDataPar->outputs_count.emplace_back(hull_x.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(hull_y.data()));
    taskDataPar->outputs_count.emplace_back(hull_y.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<beskhmelnova_k_jarvis_march_mpi::TestMPITaskParallel<double>>(taskDataPar);
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
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    int res_size = 4;
    std::vector<double> res_x = {-1.0, 1000.0, 1000.0, -1.0};
    std::vector<double> res_y = {-1.0, -1.0, 1000.0, 1000.0};
    ppc::core::Perf::print_perf_statistic(perfResults);
    for (int i = 0; i < res_size; i++) {
      EXPECT_EQ(res_x[i], hull_x[i]);
      EXPECT_EQ(res_y[i], hull_y[i]);
    }
  }
}
