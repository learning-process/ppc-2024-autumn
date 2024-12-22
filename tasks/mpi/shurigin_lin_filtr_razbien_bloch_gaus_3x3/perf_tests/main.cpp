#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/shurigin_lin_filtr_razbien_bloch_gaus_3x3/include/ops_mpi.hpp"

namespace shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi {

std::vector<int> getRandomMatrix(int rows, int cols) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<int> matrix(rows * cols);
  for (int i = 0; i < rows * cols; i++) {
    matrix[i] = dist(gen);
  }
  return matrix;
}

}  // namespace shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_rows = 1000;
  int num_cols = 1000;

  if (world.rank() == 0) {
    global_matrix = shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::getRandomMatrix(num_rows, num_cols);
    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs = {reinterpret_cast<uint8_t*>(global_matrix.data()), reinterpret_cast<uint8_t*>(&num_rows),
                           reinterpret_cast<uint8_t*>(&num_cols)};
    taskDataPar->inputs_count = {static_cast<unsigned int>(global_matrix.size()), 1u, 1u};
    taskDataPar->outputs = {reinterpret_cast<uint8_t*>(global_result.data())};
    taskDataPar->outputs_count = {static_cast<unsigned int>(global_result.size())};
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);
}

TEST(shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_rows = 1000;
  int num_cols = 1000;

  if (world.rank() == 0) {
    global_matrix = shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::getRandomMatrix(num_rows, num_cols);
    global_result.resize(num_rows * num_cols);

    taskDataPar->inputs = {reinterpret_cast<uint8_t*>(global_matrix.data()), reinterpret_cast<uint8_t*>(&num_rows),
                           reinterpret_cast<uint8_t*>(&num_cols)};
    taskDataPar->inputs_count = {static_cast<unsigned int>(global_matrix.size()), 1u, 1u};
    taskDataPar->outputs = {reinterpret_cast<uint8_t*>(global_result.data())};
    taskDataPar->outputs_count = {static_cast<unsigned int>(global_result.size())};
  }

  auto taskParallel = std::make_shared<shurigin_lin_filtr_razbien_bloch_gaus_3x3_mpi::TaskMpi>(taskDataPar);
  ASSERT_TRUE(taskParallel->validation());
  taskParallel->pre_processing();
  taskParallel->run();
  taskParallel->post_processing();

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(taskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);
}