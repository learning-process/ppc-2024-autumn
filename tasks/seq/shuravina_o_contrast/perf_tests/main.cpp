#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <chrono>
#include <iostream>
#include <vector>

#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

TEST(ParallelContrastPerf, TestContrastPerfWith1000Elements) {
  boost::mpi::communicator world;
  std::vector<uint8_t> global_vec;
  std::vector<uint8_t> global_out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 1000;
    global_vec = std::vector<uint8_t>(count_size_vector, 128);
    global_out = std::vector<uint8_t>(count_size_vector, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  shuravina_o_contrast::ContrastParallel contrastTask(taskDataPar);

  auto start_time = std::chrono::high_resolution_clock::now();
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  if (world.rank() == 0) {
    std::cout << "Execution time for 1000 elements: " << duration << " ms" << std::endl;
    for (size_t i = 0; i < global_out.size(); ++i) {
      ASSERT_EQ(global_out[i], 255);
    }
  }
}

TEST(ParallelContrastPerf, TestContrastPerfWith10000Elements) {
  boost::mpi::communicator world;
  std::vector<uint8_t> global_vec;
  std::vector<uint8_t> global_out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 10000;
    global_vec = std::vector<uint8_t>(count_size_vector, 128);
    global_out = std::vector<uint8_t>(count_size_vector, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  shuravina_o_contrast::ContrastParallel contrastTask(taskDataPar);

  auto start_time = std::chrono::high_resolution_clock::now();
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  if (world.rank() == 0) {
    std::cout << "Execution time for 10000 elements: " << duration << " ms" << std::endl;
    for (size_t i = 0; i < global_out.size(); ++i) {
      ASSERT_EQ(global_out[i], 255);
    }
  }
}

TEST(ParallelContrastPerf, TestContrastPerfWith100000Elements) {
  boost::mpi::communicator world;
  std::vector<uint8_t> global_vec;
  std::vector<uint8_t> global_out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 100000;
    global_vec = std::vector<uint8_t>(count_size_vector, 128);
    global_out = std::vector<uint8_t>(count_size_vector, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  shuravina_o_contrast::ContrastParallel contrastTask(taskDataPar);

  auto start_time = std::chrono::high_resolution_clock::now();
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  if (world.rank() == 0) {
    std::cout << "Execution time for 100000 elements: " << duration << " ms" << std::endl;
    for (size_t i = 0; i < global_out.size(); ++i) {
      ASSERT_EQ(global_out[i], 255);
    }
  }
}

TEST(ParallelContrastPerf, TestContrastPerfWith1000000Elements) {
  boost::mpi::communicator world;
  std::vector<uint8_t> global_vec;
  std::vector<uint8_t> global_out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 1000000;
    global_vec = std::vector<uint8_t>(count_size_vector, 128);
    global_out = std::vector<uint8_t>(count_size_vector, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t*>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t*>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  shuravina_o_contrast::ContrastParallel contrastTask(taskDataPar);

  auto start_time = std::chrono::high_resolution_clock::now();
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();
  auto end_time = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  if (world.rank() == 0) {
    std::cout << "Execution time for 1000000 elements: " << duration << " ms" << std::endl;
    for (size_t i = 0; i < global_out.size(); ++i) {
      ASSERT_EQ(global_out[i], 255);
    }
  }
}