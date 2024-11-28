#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <vector>

#include "mpi/shuravina_o_contrast/include/ops_mpi.hpp"

TEST(shuravina_o_contrast, Test_Contrast_10) {
  boost::mpi::communicator world;
  std::vector<uint8_t> global_vec;
  std::vector<uint8_t> global_out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 10;
    global_vec = std::vector<uint8_t>(count_size_vector, 128);
    global_out = std::vector<uint8_t>(count_size_vector, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  shuravina_o_contrast::ContrastParallel contrastTask(taskDataPar);
  ASSERT_EQ(contrastTask.validation(), true);
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();

  if (world.rank() == 0) {
    for (std::size_t i = 0; i < global_out.size(); ++i) {
      std::cout << "Output[" << i << "]: " << static_cast<int>(global_out[i]) << std::endl;
      ASSERT_EQ(global_out[i], 255);
    }
  }
}

TEST(shuravina_o_contrast, Test_Contrast_20) {
  boost::mpi::communicator world;
  std::vector<uint8_t> global_vec;
  std::vector<uint8_t> global_out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 20;
    global_vec = std::vector<uint8_t>(count_size_vector, 64);
    global_out = std::vector<uint8_t>(count_size_vector, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  shuravina_o_contrast::ContrastParallel contrastTask(taskDataPar);
  ASSERT_EQ(contrastTask.validation(), true);
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();

  if (world.rank() == 0) {
    for (std::size_t i = 0; i < global_out.size(); ++i) {
      std::cout << "Output[" << i << "]: " << static_cast<int>(global_out[i]) << std::endl;
      ASSERT_EQ(global_out[i], 255);
    }
  }
}

TEST(shuravina_o_contrast, Test_Contrast_30) {
  boost::mpi::communicator world;
  std::vector<uint8_t> global_vec;
  std::vector<uint8_t> global_out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 30;
    global_vec = std::vector<uint8_t>(count_size_vector, 32);
    global_out = std::vector<uint8_t>(count_size_vector, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  shuravina_o_contrast::ContrastParallel contrastTask(taskDataPar);
  ASSERT_EQ(contrastTask.validation(), true);
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();

  if (world.rank() == 0) {
    for (std::size_t i = 0; i < global_out.size(); ++i) {
      std::cout << "Output[" << i << "]: " << static_cast<int>(global_out[i]) << std::endl;
      ASSERT_EQ(global_out[i], 255);
    }
  }
}

TEST(shuravina_o_contrast, Test_Contrast_40) {
  boost::mpi::communicator world;
  std::vector<uint8_t> global_vec;
  std::vector<uint8_t> global_out;
  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    const int count_size_vector = 40;
    global_vec = std::vector<uint8_t>(count_size_vector, 16);
    global_out = std::vector<uint8_t>(count_size_vector, 0);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vec.data()));
    taskDataPar->inputs_count.emplace_back(global_vec.size());
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(global_out.data()));
    taskDataPar->outputs_count.emplace_back(global_out.size());
  }

  shuravina_o_contrast::ContrastParallel contrastTask(taskDataPar);
  ASSERT_EQ(contrastTask.validation(), true);
  contrastTask.pre_processing();
  contrastTask.run();
  contrastTask.post_processing();

  if (world.rank() == 0) {
    for (std::size_t i = 0; i < global_out.size(); ++i) {
      std::cout << "Output[" << i << "]: " << static_cast<int>(global_out[i]) << std::endl;
      ASSERT_EQ(global_out[i], 255);
    }
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}