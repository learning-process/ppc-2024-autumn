#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/gnitienko_k_contrast_enhancement/include/ops_mpi.hpp"

namespace gnitienko_k_functions {
std::vector<uint8_t> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::vector<uint8_t> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = gen() % 255;
  }
  return vec;
}
}  // namespace gnitienko_k_functions

TEST(gnitienko_k_contrast_enhancement_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<uint8_t> global_vec;
  double contrast_factor = 2;
  std::vector<uint8_t> global_res;
  std::vector<uint8_t> res_seq;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int img_size = 10000000;
  if (world.rank() == 0) {
    global_vec = gnitienko_k_functions::getRandomVector(img_size);
    global_res.resize(img_size, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    res_seq.resize(img_size);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&contrast_factor));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    auto testMpiTaskSeq = std::make_shared<gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq>(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq->validation(), true);
    testMpiTaskSeq->pre_processing();
    testMpiTaskSeq->run();
    testMpiTaskSeq->post_processing();
  }

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
    ASSERT_EQ(res_seq, global_res);
  }
}

TEST(gnitienko_k_contrast_enhancement_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<uint8_t> global_vec;
  double contrast_factor = 2;
  std::vector<uint8_t> global_res;
  std::vector<uint8_t> res_seq;
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int img_size = 10000000;
  if (world.rank() == 0) {
    global_vec = gnitienko_k_functions::getRandomVector(img_size);
    global_res.resize(img_size, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(&contrast_factor));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_res.data()));
    taskDataPar->outputs_count.emplace_back(global_res.size());
  }

  auto testMpiTaskParallel = std::make_shared<gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceMPI>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    res_seq.resize(img_size);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(&contrast_factor));
    taskDataSeq->inputs_count.emplace_back(global_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(res_seq.data()));
    taskDataSeq->outputs_count.emplace_back(res_seq.size());

    auto testMpiTaskSeq = std::make_shared<gnitienko_k_contrast_enhancement_mpi::ContrastEnhanceSeq>(taskDataSeq);
    ASSERT_EQ(testMpiTaskSeq->validation(), true);
    testMpiTaskSeq->pre_processing();
    testMpiTaskSeq->run();
    testMpiTaskSeq->post_processing();
  }

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
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(res_seq, global_res);
  }
}