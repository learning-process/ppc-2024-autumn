#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/fomin_v_generalized_scatter/include/ops_mpi.hpp"

TEST(fomin_v_generalized_scatter, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_input;
  std::vector<int> local_output(10, 0);  // Adjust size as needed

  int rank = world.rank();
  int size = world.size();

  if (rank == 0) {
    int count_size = size * local_output.size();
    global_input = fomin_v_generalized_scatter::getRandomVector(count_size);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskData->inputs_count.emplace_back(global_input.size());
  }
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(local_output.data()));
  taskData->outputs_count.emplace_back(local_output.size());

  auto testParallel = std::make_shared<fomin_v_generalized_scatter::GeneralizedScatterTestParallel>(taskData);
  ASSERT_EQ(testParallel->validation(), true);
  testParallel->pre_processing();
  testParallel->run();
  testParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] {
    return current_timer.elapsed();
    ;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (rank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    // Optional verification
    std::vector<int> received_data(size * local_output.size(), 0);
    MPI_Gather(nullptr, 0, MPI_INT, received_data.data(), local_output.size(), MPI_INT, 0, world);
    // Verify the received data matches the input data
    for (int i = 0; i < size; ++i) {
      int* start = &received_data[i * local_output.size()];
      for (size_t j = 0; j < local_output.size(); ++j) {
        int expected_value = global_input[i * local_output.size() + j];
        EXPECT_EQ(start[j], expected_value);
      }
    }
  } else {
    MPI_Gather(local_output.data(), local_output.size(), MPI_INT, nullptr, 0, MPI_INT, 0, world);
  }
}

// Test for running the pipeline
TEST(fomin_v_generalized_scatter, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_input;
  std::vector<int> local_output(10, 0);  // Adjust size as needed

  int rank = world.rank();
  int size = world.size();

  if (rank == 0) {
    int count_size = size * local_output.size();
    global_input = fomin_v_generalized_scatter::getRandomVector(count_size);
  }

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_input.data()));
    taskData->inputs_count.emplace_back(global_input.size());
  }
  taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(local_output.data()));
  taskData->outputs_count.emplace_back(local_output.size());

  auto testParallel = std::make_shared<fomin_v_generalized_scatter::GeneralizedScatterTestParallel>(taskData);
  ASSERT_EQ(testParallel->validation(), true);
  testParallel->pre_processing();
  testParallel->run();
  testParallel->post_processing();

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] {
    return current_timer.elapsed();
    ;
  };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (rank == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    // Optional verification
    std::vector<int> received_data(size * local_output.size(), 0);
    MPI_Gather(nullptr, 0, MPI_INT, received_data.data(), local_output.size(), MPI_INT, 0, world);
    // Verify the received data matches the input data
    for (int i = 0; i < size; ++i) {
      int* start = &received_data[i * local_output.size()];
      for (size_t j = 0; j < local_output.size(); ++j) {
        int expected_value = global_input[i * local_output.size() + j];
        EXPECT_EQ(start[j], expected_value);
      }
    }
  } else {
    MPI_Gather(local_output.data(), local_output.size(), MPI_INT, nullptr, 0, MPI_INT, 0, world);
  }
}

// namespace fomin_v_generalized_scatter