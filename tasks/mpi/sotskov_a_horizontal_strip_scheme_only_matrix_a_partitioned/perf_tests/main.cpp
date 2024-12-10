#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <boost/serialization/map.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned/include/ops_mpi.hpp"

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_A;
  std::vector<int> global_B;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_rows;
  int num_cols;

  if (world.rank() == 0) {
    num_rows = 2048;
    num_cols = 2048;

    // Generate random matrix in column-major order
    global_A.resize(num_rows * num_cols, 0);

    // Generate random vector
    global_B.resize(num_cols * num_rows, 0);
    global_result.resize(2048 * 2048, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(num_rows);
    taskDataPar->inputs_count.emplace_back(num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(num_rows);
    taskDataPar->inputs_count.emplace_back(num_cols);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel =
      std::make_shared<sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle>(
          taskDataPar);
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

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_A.data()));
    taskDataPar->inputs_count.emplace_back(num_rows);
    taskDataPar->inputs_count.emplace_back(num_cols);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_B.data()));
    taskDataPar->inputs_count.emplace_back(num_rows);
    taskDataPar->inputs_count.emplace_back(num_cols);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential =
        std::make_shared<sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential>(
            taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result, seq_result);
  }
}

TEST(sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;

  std::vector<int> global_matrix;
  std::vector<int> global_vector;
  std::vector<int> global_result;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int num_rows;
  int num_cols;

  if (world.rank() == 0) {
    num_rows = 2048;
    num_cols = 2048;

    // Generate random matrix in column-major order
    global_matrix.resize(num_rows * num_cols, 0);

    // Generate random vector
    global_vector.resize(num_cols * 2048, 0);
    global_result.resize(num_rows * 2048, 0);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(num_rows);
    taskDataPar->inputs_count.emplace_back(num_cols);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(num_rows);
    taskDataPar->inputs_count.emplace_back(num_cols);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_result.data()));
    taskDataPar->outputs_count.emplace_back(global_result.size());
  }

  auto taskParallel =
      std::make_shared<sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskParalle>(
          taskDataPar);
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

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);

    std::vector<int> seq_result(global_result.size(), 0);

    auto taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_matrix.data()));
    taskDataPar->inputs_count.emplace_back(num_rows);
    taskDataPar->inputs_count.emplace_back(num_cols);

    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vector.data()));
    taskDataPar->inputs_count.emplace_back(num_rows);
    taskDataPar->inputs_count.emplace_back(num_cols);

    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t*>(seq_result.data()));
    taskDataSeq->outputs_count.emplace_back(seq_result.size());

    auto taskSequential =
        std::make_shared<sotskov_a_horizontal_strip_scheme_only_matrix_a_partitioned_mpi::TestMPITaskSequential>(
            taskDataSeq);
    ASSERT_TRUE(taskSequential->validation());
    taskSequential->pre_processing();
    taskSequential->run();
    taskSequential->post_processing();

    ASSERT_EQ(global_result.size(), seq_result.size());
    ASSERT_EQ(global_result, seq_result);
  }
}
