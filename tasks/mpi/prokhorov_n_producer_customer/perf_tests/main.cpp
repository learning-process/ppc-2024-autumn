#include <gtest/gtest.h>

#include <boost/mpi/timer.hpp>
#include <numeric>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "mpi/prokhorov_n_producer_customer/include/ops_mpi.hpp"

TEST(prokhorov_n_producer_customer_mpi, Test_Performance) {
  boost::mpi::communicator world;
  std::vector<int> global_vec;
  std::vector<int> global_sum;
  size_t start = 2;
  size_t end = 6;

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    for (int i = 0; i < (world.size() - 1) / 2; i++) {
      global_vec.push_back(i + 1);
    }
    global_sum = global_vec;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->inputs_count.emplace_back(start);
    taskDataPar->inputs_count.emplace_back(end);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<prokhorov_n_producer_customer_mpi::TestMPITaskParallel>(taskDataPar);

  if (world.size() < 2) {
    ASSERT_EQ(testMpiTaskParallel->validation(), false);
  } else {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
  }

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

    for (int i = 0; i < (world.size() - 1) / 2; i++) {
      ASSERT_EQ(global_vec[i], global_sum[i]);
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Performance_DataSize) {
  boost::mpi::communicator world;

  for (size_t data_size = 10; data_size <= 1000; data_size *= 10) {
    std::vector<int> global_vec(data_size);
    std::vector<int> global_sum(data_size);

    auto taskDataPar = std::make_shared<ppc::core::TaskData>();

    if (world.rank() == 0) {
      std::iota(global_vec.begin(), global_vec.end(), 1);
      global_sum = global_vec;

      taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
      taskDataPar->inputs_count.emplace_back(global_vec.size());
      taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
      taskDataPar->outputs_count.emplace_back(global_sum.size());
    }

    auto testMpiTaskParallel = std::make_shared<prokhorov_n_producer_customer_mpi::TestMPITaskParallel>(taskDataPar);

    if (world.size() >= 2) {
      ASSERT_EQ(testMpiTaskParallel->validation(), true);
      testMpiTaskParallel->pre_processing();

      boost::mpi::timer timer;
      testMpiTaskParallel->run();
      double elapsed_time = timer.elapsed();

      testMpiTaskParallel->post_processing();

      if (world.rank() == 0) {
        std::cout << "Data size: " << data_size << ", Elapsed time: " << elapsed_time << " seconds." << std::endl;
        for (size_t i = 0; i < global_vec.size(); ++i) {
          ASSERT_EQ(global_vec[i], global_sum[i]);
        }
      }
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Performance_ProcessCount) {
  boost::mpi::communicator world;

  std::vector<int> global_vec(100);
  std::vector<int> global_sum(100);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::iota(global_vec.begin(), global_vec.end(), 1);
    global_sum = global_vec;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<prokhorov_n_producer_customer_mpi::TestMPITaskParallel>(taskDataPar);

  if (world.size() >= 2) {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();

    boost::mpi::timer timer;
    testMpiTaskParallel->run();
    double elapsed_time = timer.elapsed();

    testMpiTaskParallel->post_processing();

    if (world.rank() == 0) {
      std::cout << "Process count: " << world.size() << ", Elapsed time: " << elapsed_time << " seconds." << std::endl;
      for (size_t i = 0; i < global_vec.size(); ++i) {
        ASSERT_EQ(global_vec[i], global_sum[i]);
      }
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Performance_ConcurrentAccess) {
  boost::mpi::communicator world;

  size_t buffer_capacity = 20;
  std::vector<int> global_vec(buffer_capacity * 10);
  std::vector<int> global_sum(buffer_capacity * 10);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::iota(global_vec.begin(), global_vec.end(), 1);
    global_sum = global_vec;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<prokhorov_n_producer_customer_mpi::TestMPITaskParallel>(taskDataPar);

  if (world.size() >= 2) {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();

    boost::mpi::timer timer;
    testMpiTaskParallel->run();
    double elapsed_time = timer.elapsed();

    testMpiTaskParallel->post_processing();

    if (world.rank() == 0) {
      std::cout << "Concurrent buffer access, Elapsed time: " << elapsed_time << " seconds." << std::endl;
      for (size_t i = 0; i < global_vec.size(); ++i) {
        ASSERT_EQ(global_vec[i], global_sum[i]);
      }
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Scalability_ProducersConsumers) {
  boost::mpi::communicator world;

  size_t data_size = 500;
  std::vector<int> global_vec(data_size);
  std::vector<int> global_sum(data_size);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::iota(global_vec.begin(), global_vec.end(), 1);
    global_sum = global_vec;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<prokhorov_n_producer_customer_mpi::TestMPITaskParallel>(taskDataPar);

  if (world.size() >= 2) {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();

    boost::mpi::timer timer;
    testMpiTaskParallel->run();
    double elapsed_time = timer.elapsed();

    testMpiTaskParallel->post_processing();

    if (world.rank() == 0) {
      std::cout << "Data size: " << data_size << ", Processes: " << world.size() << ", Elapsed time: " << elapsed_time
                << " seconds." << std::endl;

      for (size_t i = 0; i < global_vec.size(); ++i) {
        ASSERT_EQ(global_vec[i], global_sum[i]);
      }
    }
  }
}

TEST(prokhorov_n_producer_customer_mpi, Test_Performance_WithDelays) {
  boost::mpi::communicator world;

  size_t data_size = 100;
  std::vector<int> global_vec(data_size);
  std::vector<int> global_sum(data_size);

  auto taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    std::iota(global_vec.begin(), global_vec.end(), 1);
    global_sum = global_vec;

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_sum.data()));
    taskDataPar->outputs_count.emplace_back(global_sum.size());
  }

  auto testMpiTaskParallel = std::make_shared<prokhorov_n_producer_customer_mpi::TestMPITaskParallel>(taskDataPar);

  if (world.size() >= 2) {
    ASSERT_EQ(testMpiTaskParallel->validation(), true);
    testMpiTaskParallel->pre_processing();

    boost::mpi::timer timer;
    testMpiTaskParallel->run();
    double elapsed_time = timer.elapsed();

    testMpiTaskParallel->post_processing();

    if (world.rank() == 0) {
      std::cout << "Data size: " << data_size << ", Elapsed time with delays: " << elapsed_time << " seconds."
                << std::endl;

      for (size_t i = 0; i < global_vec.size(); ++i) {
        ASSERT_EQ(global_vec[i], global_sum[i]);
      }
    }
  }
}
