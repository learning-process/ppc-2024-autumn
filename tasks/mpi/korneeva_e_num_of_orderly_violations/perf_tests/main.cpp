#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/korneeva_e_num_of_orderly_violations/include/ops_mpi.hpp"

// Test to measure and validate pipeline execution
TEST(korneeva_e_num_of_orderly_violations_mpi, test_pipeline_execution) {
  const int vector_size = 10000000;           // Define vector size
  std::vector<int> data_vector(vector_size);  // Data vector
  std::vector<int32_t> result_buffer(1, 0);   // Result buffer
  boost::mpi::communicator comm_world;        // Create MPI communicator

  // Create task data
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (comm_world.rank() == 0) {
    std::random_device random_device;
    std::default_random_engine engine(random_device());
    std::uniform_int_distribution<int> distribution(0, 100);  // Generate random values from 0 to 100
    std::generate(data_vector.begin(), data_vector.end(), [&]() { return distribution(engine); });

    // Fill input and output data
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    task_data->inputs_count.emplace_back(data_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_buffer.data()));
    task_data->outputs_count.emplace_back(result_buffer.size());
  }

  // Create MPI task
  auto mpi_task =
      std::make_shared<korneeva_e_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int>>(task_data);

  // Setup performance attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;                                          // Number of runs for performance measurement
  const boost::mpi::timer timer_instance;                              // Timer instance
  perfAttr->current_timer = [&] { return timer_instance.elapsed(); };  // Lambda for elapsed time

  // Initialize performance results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create performance analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);  // Run the pipeline

  if (comm_world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);                      // Print performance statistics
    auto computed_result = mpi_task->count_orderly_violations(data_vector);  // Sequential processing
    ASSERT_EQ(computed_result, result_buffer[0]);                            // Validate result
  }
}

// Test to measure and validate task execution
TEST(korneeva_e_num_of_orderly_violations_mpi, test_task_run) {
  const int numElements = 10000000;  // Number of elements in the vector
  boost::mpi::communicator world;
  std::vector<int> global_vec(numElements);  // Global vector
  std::vector<int32_t> out(1, 0);            // Output buffer

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device randomDevice;
    std::default_random_engine reng(randomDevice());
    std::uniform_int_distribution<int> dist(0, 100);  // Generate values between 0 and 100
    std::generate(global_vec.begin(), global_vec.end(), [&dist, &reng] { return dist(reng); });
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskData->inputs_count.emplace_back(global_vec.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  // Create MPI task
  auto testMpiTaskParallel =
      std::make_shared<korneeva_e_num_of_orderly_violations_mpi::num_of_orderly_violations<int, int>>(taskData);

  // Create a buffer for receiving data
  std::vector<int> recv_buf(numElements / world.size());  // Size adjusted for scatter logic

  // Create performance attributes
  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;                                         // Number of runs for performance measurement
  const boost::mpi::timer current_timer;                              // Timer instance
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };  // Lambda for elapsed time

  // Create and initialize performance results
  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  // Create performance analyzer
  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);

  // Run the task
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);  // Print performance statistics
    // Count orderly violations on the master process
    auto temp = testMpiTaskParallel->count_orderly_violations(global_vec);
    ASSERT_EQ(out[0], temp);  // Validate result
  }
}
