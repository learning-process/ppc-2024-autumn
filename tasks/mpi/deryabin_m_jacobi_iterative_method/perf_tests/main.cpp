#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>

#include "core/perf/include/perf.hpp"
#include "mpi/deryabin_m_jacobi_iterative_method/include/ops_mpi.hpp"

TEST(deryabin_m_jacobi_iterative_method_mpi, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_ = std::vector<double>(10000);
  std::vector<double> input_right_vector_ = std::vector<double>(100);
  std::vector<double> output_x_vector_ = std::vector<double>(100, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);
  for (unsigned short razmernost = 0; razmernost < 10000; razmernost++) {
    if (razmernost < 100) {
      input_right_vector_[razmernost] = razmernost + 1;
    }
    if (razmernost % 101 == 0) {
      input_matrix_[razmernost] = 1;
    } else {
      input_matrix_[razmernost] = 0;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel>(taskDataPar);
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
    ASSERT_EQ(input_right_vector_, out_x_vec[0]);
  }
}

TEST(deryabin_m_jacobi_iterative_method_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<double> input_matrix_ = std::vector<double>(10000);
  std::vector<double> input_right_vector_ = std::vector<double>(100);
  std::vector<double> output_x_vector_ = std::vector<double>(100, 0);
  std::vector<std::vector<double>> out_x_vec(1, output_x_vector_);
  for (unsigned short razmernost = 0; razmernost < 10000; razmernost++) {
    if (razmernost < 100) {
      input_right_vector_[razmernost] = razmernost + 1;
    }
    if (razmernost % 101 == 0) {
      input_matrix_[razmernost] = 1;
    } else {
      input_matrix_[razmernost] = 0;
    }
  }
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_matrix_.data()));
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_right_vector_.data()));
    taskDataPar->inputs_count.emplace_back(input_right_vector_.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(out_x_vec.data()));
    taskDataPar->outputs_count.emplace_back(out_x_vec.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<deryabin_m_jacobi_iterative_method_mpi::JacobiIterativeMPITaskParallel>(taskDataPar);
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
  perfAnalyzer->task_run(perfAttr, perfResults);
  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    ASSERT_EQ(input_right_vector_, out_x_vec[0]);
  }
}
