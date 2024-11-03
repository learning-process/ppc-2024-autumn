#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/vavilov_v_min_elements_in_columns_of_matrix/include/ops_mpi.hpp"

TEST(vavilov_v_min_elements_in_columns_of_matrix_mpi, test_pipeline_run_min) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> min_col;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_row;
  int count_col;

  if (world.rank() == 0) {
    count_row = 5000;
    count_col = 5000;
    global_matr = vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential::generate_rand_matr(count_row,
                                                                                                             count_col);
    min_col.resize(count_col, INT_MAX);

    for (auto& row : global_matr) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }
    taskDataPar->inputs_count.emplace_back(count_row);
    taskDataPar->inputs_count.emplace_back(count_col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(min_col.data()));
    taskDataPar->outputs_count.emplace_back(min_col.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    for (size_t j = 0; j < min_col.size(); j++) {
      ASSERT_EQ(min_col[j], INT_MIN);
    }
  }
}

TEST(vavilov_v_min_elements_in_columns_of_matrix_mpi, test_task_run) {
  boost::mpi::communicator world;
  std::vector<std::vector<int>> global_matr;
  std::vector<int32_t> min_col;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  int count_row;
  int count_col;

  if (world.rank() == 0) {
    count_row = 5000;
    count_col = 5000;
    global_matr = vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskSequential::generate_rand_matr(count_row,
                                                                                                             count_col);
    min_col.resize(count_col, INT_MAX);

    for (auto& row : global_matr) {
      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(row.data()));
    }
    taskDataPar->inputs_count.emplace_back(count_row);
    taskDataPar->inputs_count.emplace_back(count_col);

    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(min_col.data()));
    taskDataPar->outputs_count.emplace_back(min_col.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<vavilov_v_min_elements_in_columns_of_matrix_mpi::TestMPITaskParallel>(taskDataPar);
  ASSERT_EQ(testMpiTaskParallel->validation(), true);
  testMpiTaskParallel->pre_processing();
  testMpiTaskParallel->run();
  testMpiTaskParallel->post_processing();

  if (world.rank() == 0) {
    for (size_t j = 0; j < min_col.size(); j++) {
      ASSERT_EQ(min_col[j], INT_MIN);
    }
  }
}
