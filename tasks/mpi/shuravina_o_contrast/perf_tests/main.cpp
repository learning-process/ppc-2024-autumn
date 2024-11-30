#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

TEST(shuravina_o_contrast, Test_Contrast_Small) {
  boost::mpi::communicator world;
  const int count = 100;

  std::vector<uint8_t> in(count);
  std::vector<uint8_t> out(count, 0);

  if (world.rank() == 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < count; ++i) {
      in[i] = dis(gen);
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto contrastTaskParallel = std::make_shared<shuravina_o_contrast::ContrastTaskParallel>(taskDataPar);
  ASSERT_EQ(contrastTaskParallel->validation(), true);
  contrastTaskParallel->pre_processing();
  contrastTaskParallel->run();
  contrastTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(contrastTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
TEST(shuravina_o_contrast, Test_Contrast_Medium) {
  boost::mpi::communicator world;
  const int count = 10000;

  std::vector<uint8_t> in(count);
  std::vector<uint8_t> out(count, 0);

  if (world.rank() == 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < count; ++i) {
      in[i] = dis(gen);
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto contrastTaskParallel = std::make_shared<shuravina_o_contrast::ContrastTaskParallel>(taskDataPar);
  ASSERT_EQ(contrastTaskParallel->validation(), true);
  contrastTaskParallel->pre_processing();
  contrastTaskParallel->run();
  contrastTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(contrastTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
TEST(shuravina_o_contrast, Test_Contrast_Large) {
  boost::mpi::communicator world;
  const int count = 1000000;

  std::vector<uint8_t> in(count);
  std::vector<uint8_t> out(count, 0);

  if (world.rank() == 0) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    for (int i = 0; i < count; ++i) {
      in[i] = dis(gen);
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto contrastTaskParallel = std::make_shared<shuravina_o_contrast::ContrastTaskParallel>(taskDataPar);
  ASSERT_EQ(contrastTaskParallel->validation(), true);
  contrastTaskParallel->pre_processing();
  contrastTaskParallel->run();
  contrastTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(contrastTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
TEST(shuravina_o_contrast, Test_Contrast_Uniform) {
  boost::mpi::communicator world;
  const int count = 10000;

  std::vector<uint8_t> in(count, 128);
  std::vector<uint8_t> out(count, 0);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto contrastTaskParallel = std::make_shared<shuravina_o_contrast::ContrastTaskParallel>(taskDataPar);
  ASSERT_EQ(contrastTaskParallel->validation(), true);
  contrastTaskParallel->pre_processing();
  contrastTaskParallel->run();
  contrastTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(contrastTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}
TEST(shuravina_o_contrast, Test_Contrast_MinMax) {
  boost::mpi::communicator world;
  const int count = 10000;

  std::vector<uint8_t> in(count);
  std::vector<uint8_t> out(count, 0);

  if (world.rank() == 0) {
    for (int i = 0; i < count; ++i) {
      in[i] = (i % 2 == 0) ? 0 : 255;
    }
  }

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(in.data()));
    taskDataPar->inputs_count.emplace_back(in.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskDataPar->outputs_count.emplace_back(out.size());
  }

  auto contrastTaskParallel = std::make_shared<shuravina_o_contrast::ContrastTaskParallel>(taskDataPar);
  ASSERT_EQ(contrastTaskParallel->validation(), true);
  contrastTaskParallel->pre_processing();
  contrastTaskParallel->run();
  contrastTaskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(contrastTaskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
  }
}