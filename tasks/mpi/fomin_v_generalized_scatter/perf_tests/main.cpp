#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/fomin_v_generalized_scatter/include/ops_mpi.hpp"

TEST(fomin_v_generalized_scatter, test_task_run) {
  boost::mpi::communicator world;
  std::vector<int> global_input;
  std::vector<int> local_output(10, 0);  // Adjust size as needed

  // Create TaskData
  std::shared_ptr<TaskData> taskData = std::make_shared<TaskData>();
  int count_size;
  if (world.rank() == 0) {
    count_size = 100;  // Adjust size as needed
    global_input = getRandomVector(count_size);
    taskData->inputs[0] = global_input.data();
    taskData->inputs_count[0] = global_input.size();
    taskData->outputs[0] = local_output.data();
    taskData->outputs_count[0] = local_output.size();
    taskData->datatype = MPI_INT;
    taskData->ops = "scatter";  // Example operation
  }

  auto testParallel = std::make_shared<GeneralizedScatterTestParallel>();
  ASSERT_EQ(testParallel->validation(taskData.get()), true);
  testParallel->pre_processing(taskData.get());
  testParallel->run(taskData.get());
  testParallel->post_processing(taskData.get());

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Number of runs for performance measurement
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    // Add assertions if needed to validate the output
  }
}

// Test for running the pipeline
TEST(fomin_v_generalized_scatter, test_pipeline_run) {
  boost::mpi::communicator world;
  std::vector<int> global_input;
  std::vector<int> local_output(10, 0);  // Adjust size as needed

  // Create TaskData
  std::shared_ptr<TaskData> taskData = std::make_shared<TaskData>();
  int count_size;
  if (world.rank() == 0) {
    count_size = 100;  // Adjust size as needed
    global_input = getRandomVector(count_size);
    taskData->inputs[0] = global_input.data();
    taskData->inputs_count[0] = global_input.size();
    taskData->outputs[0] = local_output.data();
    taskData->outputs_count[0] = local_output.size();
    taskData->datatype = MPI_INT;
    taskData->ops = "scatter";  // Example operation
  }

  auto testParallel = std::make_shared<GeneralizedScatterTestParallel>();
  ASSERT_EQ(testParallel->validation(taskData.get()), true);
  testParallel->pre_processing(taskData.get());
  testParallel->run(taskData.get());
  testParallel->post_processing(taskData.get());

  // Create Perf attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;  // Number of runs for performance measurement
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  // Create and init perf results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testParallel);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    // Add assertions if needed to validate the output
  }
}

// namespace fomin_v_generalized_scatter