#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/koshkin_n_linear_histogram_stretch/include/ops_mpi.hpp"

TEST(koshkin_n_linear_histogram_stretch_mpi, test_pipeline_run) {
  boost::mpi::communicator world;

  const int width = 173;
  const int height = 173;

  std::vector<int> in_vec;
  const int count_size_vector = width * height * 3;
  std::vector<int> out_vec_par(count_size_vector, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in_vec = koshkin_n_linear_histogram_stretch_mpi::getRandomImage(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
  }

  auto testMpiTaskParallel = std::make_shared<koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel>(taskDataPar);
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

    std::vector<int> out_vec_seq(count_size_vector, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataSeq->inputs_count.emplace_back(in_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_vec_seq.size());

    // Create Task
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}

TEST(koshkin_n_linear_histogram_stretch_mpi, test_task_run) {
  boost::mpi::communicator world;

  const int width = 173;
  const int height = 173;

  std::vector<int> in_vec;
  const int count_size_vector = width * height * 3;
  std::vector<int> out_vec_par(count_size_vector, 0);
  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in_vec = koshkin_n_linear_histogram_stretch_mpi::getRandomImage(count_size_vector);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataPar->inputs_count.emplace_back(in_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_par.data()));
    taskDataPar->outputs_count.emplace_back(out_vec_par.size());
  }

  auto testMpiTaskParallel = std::make_shared<koshkin_n_linear_histogram_stretch_mpi::TestMPITaskParallel>(taskDataPar);
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
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> out_vec_seq(count_size_vector, 0);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in_vec.data()));
    taskDataSeq->inputs_count.emplace_back(in_vec.size());
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_vec_seq.data()));
    taskDataSeq->outputs_count.emplace_back(out_vec_seq.size());

    // Create Task
    koshkin_n_linear_histogram_stretch_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();

    ASSERT_EQ(out_vec_par, out_vec_seq);
  }
}