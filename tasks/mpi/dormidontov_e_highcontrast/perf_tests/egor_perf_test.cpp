#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/dormidontov_e_highcontrast/include/egor_include.hpp"

namespace dormidontov_e_highcontrast_mpi {
std::vector<int> generate_pic(int heigh, int width) {
  std::vector<int> tmp(heigh * width);
  for (int i = 0; i < heigh; ++i) {
    for (int j = 0; j < width; ++j) {
      tmp[i * width + j] = (i * width + j) % 6;
    }
  }
  return tmp;
}
std::vector<int> generate_answer(int heigh, int width) {
  std::vector<int> tmp(heigh * width);
  for (int i = 0; i < heigh; ++i) {
    for (int j = 0; j < width; ++j) {
      tmp[i * width + j] = ((i * width + j) % 6) * 51;
    }
  }
  return tmp;
}
}  // namespace dormidontov_e_highcontrast_mpi

TEST(dormidontov_e_highcontrast_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  int height = 1984;
  int width = 1984;

  std::vector<int> picture = dormidontov_e_highcontrast_mpi::generate_pic(height, width);
  std::vector<int> res_out_paral(width * height, 0);
  std::vector<int> exp_res_paral = dormidontov_e_highcontrast_mpi::generate_answer(height, width);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataPar->inputs_count.emplace_back(height * width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  auto ContrastP = std::make_shared<dormidontov_e_highcontrast_mpi::ContrastP>(taskDataPar);
  ASSERT_EQ(ContrastP->validation(), true);
  ContrastP->pre_processing();
  ContrastP->run();
  ContrastP->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ContrastP);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(res_out_paral, exp_res_paral);
  }
}

TEST(dormidontov_e_highcontrast_mpi, test_task_run) {
  boost::mpi::communicator world;
  int height = 1984;
  int width = 1984;

  std::vector<int> picture = dormidontov_e_highcontrast_mpi::generate_pic(height, width);
  std::vector<int> res_out_paral(width * height, 0);
  std::vector<int> exp_res_paral = dormidontov_e_highcontrast_mpi::generate_answer(height, width);

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(picture.data()));
    taskDataPar->inputs_count.emplace_back(height * width);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(res_out_paral.data()));
    taskDataPar->outputs_count.emplace_back(res_out_paral.size());
  }
  auto ContrastP = std::make_shared<dormidontov_e_highcontrast_mpi::ContrastP>(taskDataPar);
  ASSERT_EQ(ContrastP->validation(), true);
  ContrastP->pre_processing();
  ContrastP->run();
  ContrastP->post_processing();
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };
  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(ContrastP);
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(res_out_paral, exp_res_paral);
  }
}