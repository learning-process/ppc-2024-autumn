#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/timer.hpp>
#include <random>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/somov_i_horizontal_scheme/include/ops_mpi.hpp"

namespace somov_i_horizontal_scheme {

std::vector<int32_t> create_random_vector(uint32_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 300.0f);

  std::vector<int32_t> vec(size);
  for (auto &el : vec) {
    el = std::clamp(static_cast<int32_t>(std::round(dist(gen))), -900, 900);
  }
  return vec;
}

std::vector<int32_t> create_random_matrix(uint32_t rowCount, uint32_t colCount) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 300.0f);

  std::vector<int32_t> matrix(rowCount * colCount);
  for (auto &el : matrix) {
    el = std::clamp(static_cast<int32_t>(std::round(dist(gen))), -900, 900);
  }
  return matrix;
}

}  // namespace somov_i_horizontal_scheme

TEST(somov_i_horizontal_scheme, test_pipeline_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  std::vector<int32_t> matrix_data;
  std::vector<int32_t> input_vector;
  std::vector<int32_t> result_vector;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  int num_rows;
  int num_cols;

  if (world.rank() == 0) {
    num_rows = 2000;
    num_cols = 2000;
    matrix_data = somov_i_horizontal_scheme::create_random_matrix(num_rows, num_cols);
    input_vector = somov_i_horizontal_scheme::create_random_vector(num_cols);
    result_vector.resize(num_rows, 0);

    task_data->inputs.push_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    task_data->inputs_count.push_back(matrix_data.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data->inputs_count.push_back(input_vector.size());
    task_data->outputs.push_back(reinterpret_cast<uint8_t *>(result_vector.data()));
    task_data->outputs_count.push_back(result_vector.size());
  }

  auto parallel_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTaskMPI>(task_data);

  parallel_task->validation();
  parallel_task->pre_processing();
  parallel_task->run();
  parallel_task->post_processing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;

  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(parallel_task);
  perf_analyzer->pipeline_run(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);

    std::vector<int32_t> seq_result(result_vector.size(), 0);

    auto sequential_task_data = std::make_shared<ppc::core::TaskData>();
    sequential_task_data->inputs.push_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    sequential_task_data->inputs_count.push_back(matrix_data.size());
    sequential_task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    sequential_task_data->inputs_count.push_back(input_vector.size());
    sequential_task_data->outputs.push_back(reinterpret_cast<uint8_t *>(seq_result.data()));
    sequential_task_data->outputs_count.push_back(seq_result.size());

    auto sequential_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(sequential_task_data);
    sequential_task->validation();
    sequential_task->pre_processing();
    sequential_task->run();
    sequential_task->post_processing();

    ASSERT_EQ(result_vector.size(), seq_result.size());
    for (size_t i = 0; i < result_vector.size(); ++i) {
      ASSERT_EQ(result_vector[i], seq_result[i]);
    }
  }
}

TEST(somov_i_horizontal_scheme, test_task_run) {
  boost::mpi::environment env;
  boost::mpi::communicator world;
  std::vector<int32_t> matrix_data;
  std::vector<int32_t> input_vector;
  std::vector<int32_t> result_vector;
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  int num_rows;
  int num_cols;

  if (world.rank() == 0) {
    num_rows = 2001;
    num_cols = 1999;
    matrix_data = somov_i_horizontal_scheme::create_random_matrix(num_rows, num_cols);
    input_vector = somov_i_horizontal_scheme::create_random_vector(num_cols);
    result_vector.resize(num_rows, 0);

    task_data->inputs.push_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    task_data->inputs_count.push_back(matrix_data.size());
    task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    task_data->inputs_count.push_back(input_vector.size());
    task_data->outputs.push_back(reinterpret_cast<uint8_t *>(result_vector.data()));
    task_data->outputs_count.push_back(result_vector.size());
  }

  auto parallel_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTaskMPI>(task_data);

  parallel_task->validation();
  parallel_task->pre_processing();
  parallel_task->run();
  parallel_task->post_processing();

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;

  const boost::mpi::timer current_timer;
  perf_attr->current_timer = [&] { return current_timer.elapsed(); };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(parallel_task);
  perf_analyzer->task_run(perf_attr, perf_results);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perf_results);

    std::vector<int32_t> seq_result(result_vector.size(), 0);

    auto sequential_task_data = std::make_shared<ppc::core::TaskData>();
    sequential_task_data->inputs.push_back(reinterpret_cast<uint8_t *>(matrix_data.data()));
    sequential_task_data->inputs_count.push_back(matrix_data.size());
    sequential_task_data->inputs.push_back(reinterpret_cast<uint8_t *>(input_vector.data()));
    sequential_task_data->inputs_count.push_back(input_vector.size());
    sequential_task_data->outputs.push_back(reinterpret_cast<uint8_t *>(seq_result.data()));
    sequential_task_data->outputs_count.push_back(seq_result.size());

    auto sequential_task = std::make_shared<somov_i_horizontal_scheme::MatrixVectorTask>(sequential_task_data);
    sequential_task->validation();
    sequential_task->pre_processing();
    sequential_task->run();
    sequential_task->post_processing();

    ASSERT_EQ(result_vector.size(), seq_result.size());
    for (size_t i = 0; i < result_vector.size(); ++i) {
      ASSERT_EQ(result_vector[i], seq_result[i]);
    }
  }
}
