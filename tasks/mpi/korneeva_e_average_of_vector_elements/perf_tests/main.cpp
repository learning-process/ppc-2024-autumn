#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <random>

#include "core/perf/include/perf.hpp"
#include "mpi/korneeva_e_average_of_vector_elements/include/ops_mpi.hpp"

// Tests to measure and validate pipeline execution
TEST(korneeva_e_average_of_vector_elements_mpi, test_pipeline_execution_int_MPI_AllReduce) {
  const int vector_size = 10000000;
  std::vector<int> data_vector(vector_size);
  std::vector<double> result_buffer(1, 0.0);
  boost::mpi::communicator comm_world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (comm_world.rank() == 0) {
    std::random_device random_device;
    std::default_random_engine engine(random_device());
    std::uniform_int_distribution<int> distribution(0, 100);
    std::generate(data_vector.begin(), data_vector.end(), [&]() { return distribution(engine); });

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    task_data->inputs_count.emplace_back(data_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_buffer.data()));
    task_data->outputs_count.emplace_back(result_buffer.size());
  }

  auto mpi_task =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer timer_instance;
  perfAttr->current_timer = [&] { return timer_instance.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (comm_world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(computed_result, result_buffer[0], 1e-5);
  }
}

TEST(korneeva_e_average_of_vector_elements_mpi, test_pipeline_execution_int_my_AllReduce) {
  const int vector_size = 10000000;
  std::vector<int> data_vector(vector_size);
  std::vector<double> result_buffer(1, 0.0);
  boost::mpi::communicator comm_world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (comm_world.rank() == 0) {
    std::random_device random_device;
    std::default_random_engine engine(random_device());
    std::uniform_int_distribution<int> distribution(0, 100);
    std::generate(data_vector.begin(), data_vector.end(), [&]() { return distribution(engine); });

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    task_data->inputs_count.emplace_back(data_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_buffer.data()));
    task_data->outputs_count.emplace_back(result_buffer.size());
  }

  auto mpi_task =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer timer_instance;
  perfAttr->current_timer = [&] { return timer_instance.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (comm_world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(computed_result, result_buffer[0], 1e-5);
  }
}

TEST(korneeva_e_average_of_vector_elements_mpi, test_pipeline_execution_float_MPI_AllReduce) {
  const int vector_size = 10000000;
  std::vector<float> data_vector(vector_size);
  std::vector<double> result_buffer(1, 0.0);
  boost::mpi::communicator comm_world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (comm_world.rank() == 0) {
    std::random_device random_device;
    std::default_random_engine engine(random_device());
    std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
    std::generate(data_vector.begin(), data_vector.end(), [&]() { return distribution(engine); });

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    task_data->inputs_count.emplace_back(data_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_buffer.data()));
    task_data->outputs_count.emplace_back(result_buffer.size());
  }

  auto mpi_task =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<float>>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer timer_instance;
  perfAttr->current_timer = [&] { return timer_instance.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (comm_world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(computed_result, result_buffer[0], 1e-6);
  }
}
TEST(korneeva_e_average_of_vector_elements_mpi, test_pipeline_execution_float_my_AllReduce) {
  const int vector_size = 10000000;
  std::vector<float> data_vector(vector_size);
  std::vector<double> result_buffer(1, 0.0);
  boost::mpi::communicator comm_world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (comm_world.rank() == 0) {
    std::random_device random_device;
    std::default_random_engine engine(random_device());
    std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
    std::generate(data_vector.begin(), data_vector.end(), [&]() { return distribution(engine); });

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    task_data->inputs_count.emplace_back(data_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_buffer.data()));
    task_data->outputs_count.emplace_back(result_buffer.size());
  }

  auto mpi_task =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<float>>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer timer_instance;
  perfAttr->current_timer = [&] { return timer_instance.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (comm_world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(computed_result, result_buffer[0], 1e-6);
  }
}

TEST(korneeva_e_average_of_vector_elements_mpi, test_pipeline_execution_double_MPI_AllReduce) {
  const int vector_size = 10000000;
  std::vector<double> data_vector(vector_size);
  std::vector<double> result_buffer(1, 0.0);
  boost::mpi::communicator comm_world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (comm_world.rank() == 0) {
    std::random_device random_device;
    std::default_random_engine engine(random_device());
    std::uniform_real_distribution<double> distribution(0.0, 100.0);
    std::generate(data_vector.begin(), data_vector.end(), [&]() { return distribution(engine); });

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    task_data->inputs_count.emplace_back(data_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_buffer.data()));
    task_data->outputs_count.emplace_back(result_buffer.size());
  }

  auto mpi_task =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<double>>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer timer_instance;
  perfAttr->current_timer = [&] { return timer_instance.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (comm_world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(computed_result, result_buffer[0], 1e-6);
  }
}
TEST(korneeva_e_average_of_vector_elements_mpi, test_pipeline_execution_double_my_AllReduce) {
  const int vector_size = 10000000;
  std::vector<double> data_vector(vector_size);
  std::vector<double> result_buffer(1, 0.0);
  boost::mpi::communicator comm_world;

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (comm_world.rank() == 0) {
    std::random_device random_device;
    std::default_random_engine engine(random_device());
    std::uniform_real_distribution<double> distribution(0.0, 100.0);
    std::generate(data_vector.begin(), data_vector.end(), [&]() { return distribution(engine); });

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    task_data->inputs_count.emplace_back(data_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_buffer.data()));
    task_data->outputs_count.emplace_back(result_buffer.size());
  }

  auto mpi_task =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<double>>(task_data);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer timer_instance;
  perfAttr->current_timer = [&] { return timer_instance.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perfAnalyzer->pipeline_run(perfAttr, perfResults);

  if (comm_world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(computed_result, result_buffer[0], 1e-6);
  }
}
// Tests to measure and validate task execution
TEST(korneeva_e_average_of_vector_elements_mpi, test_task_run_int_MPI_AllReduce) {
  const int numElements = 10000000;
  boost::mpi::communicator world;
  std::vector<int> data_vector(numElements);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device randomDevice;
    std::default_random_engine reng(randomDevice());
    std::uniform_int_distribution<int> dist(0, 100);
    std::generate(data_vector.begin(), data_vector.end(), [&dist, &reng] { return dist(reng); });
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    taskData->inputs_count.emplace_back(data_vector.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<int>>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(out[0], computed_result, 1e-6);
  }
}
TEST(korneeva_e_average_of_vector_elements_mpi, test_task_run_int_my_AllReduce) {
  const int numElements = 10000000;
  boost::mpi::communicator world;
  std::vector<int> data_vector(numElements);
  std::vector<double> out(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> taskData = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::random_device randomDevice;
    std::default_random_engine reng(randomDevice());
    std::uniform_int_distribution<int> dist(0, 100);
    std::generate(data_vector.begin(), data_vector.end(), [&dist, &reng] { return dist(reng); });
    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    taskData->inputs_count.emplace_back(data_vector.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<int>>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);

  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(out[0], computed_result, 1e-6);
  }
}

TEST(korneeva_e_average_of_vector_elements_mpi, test_task_run_float_MPI_AllReduce) {
  const int numElements = 10000000;
  boost::mpi::communicator world;
  std::vector<float> data_vector(numElements);
  std::vector<double> out(1, 0.0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device randomDevice;
    std::default_random_engine reng(randomDevice());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    std::generate(data_vector.begin(), data_vector.end(), [&dist, &reng] { return dist(reng); });

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    taskData->inputs_count.emplace_back(data_vector.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<float>>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(out[0], computed_result, 1e-6);
  }
}
TEST(korneeva_e_average_of_vector_elements_mpi, test_task_run_float_my_AllReduce) {
  const int numElements = 10000000;
  boost::mpi::communicator world;
  std::vector<float> data_vector(numElements);
  std::vector<double> out(1, 0.0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device randomDevice;
    std::default_random_engine reng(randomDevice());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    std::generate(data_vector.begin(), data_vector.end(), [&dist, &reng] { return dist(reng); });

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    taskData->inputs_count.emplace_back(data_vector.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<float>>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(out[0], computed_result, 1e-6);
  }
}

TEST(korneeva_e_average_of_vector_elements_mpi, test_task_run_double_MPI_AllReduce) {
  const int numElements = 10000000;
  boost::mpi::communicator world;
  std::vector<double> data_vector(numElements);
  std::vector<double> out(1, 0.0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device randomDevice;
    std::default_random_engine reng(randomDevice());
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    std::generate(data_vector.begin(), data_vector.end(), [&dist, &reng] { return dist(reng); });

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    taskData->inputs_count.emplace_back(data_vector.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_MPI_AllReduce<double>>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(out[0], computed_result, 1e-6);
  }
}

TEST(korneeva_e_average_of_vector_elements_mpi, test_task_run_double_my_AllReduce) {
  const int numElements = 10000000;
  boost::mpi::communicator world;
  std::vector<double> data_vector(numElements);
  std::vector<double> out(1, 0.0);

  auto taskData = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::random_device randomDevice;
    std::default_random_engine reng(randomDevice());
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    std::generate(data_vector.begin(), data_vector.end(), [&dist, &reng] { return dist(reng); });

    taskData->inputs.emplace_back(reinterpret_cast<uint8_t*>(data_vector.data()));
    taskData->inputs_count.emplace_back(data_vector.size());
    taskData->outputs.emplace_back(reinterpret_cast<uint8_t*>(out.data()));
    taskData->outputs_count.emplace_back(out.size());
  }

  auto testMpiTaskParallel =
      std::make_shared<korneeva_e_average_of_vector_elements_mpi::vector_average_my_AllReduce<double>>(taskData);

  auto perfAttr = std::make_shared<ppc::core::PerfAttr>();
  perfAttr->num_running = 10;
  const boost::mpi::timer current_timer;
  perfAttr->current_timer = [&] { return current_timer.elapsed(); };

  auto perfResults = std::make_shared<ppc::core::PerfResults>();

  auto perfAnalyzer = std::make_shared<ppc::core::Perf>(testMpiTaskParallel);
  perfAnalyzer->task_run(perfAttr, perfResults);

  if (world.rank() == 0) {
    ppc::core::Perf::print_perf_statistic(perfResults);
    auto computed_result = std::accumulate(data_vector.begin(), data_vector.end(), 0.0) / data_vector.size();
    ASSERT_NEAR(out[0], computed_result, 1e-6);
  }
}
