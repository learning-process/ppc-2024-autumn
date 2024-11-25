// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mpi/beresnev_a_increase_contrast/include/ops_mpi.hpp"

static std::vector<uint8_t> getRandomVector(int sz) {
  std::random_device dev;
  std::mt19937 gen(dev());
  std::uniform_int_distribution<int> dist(0, 255);
  std::vector<uint8_t> vec(sz);
  for (int i = 0; i < sz; i++) {
    vec[i] = dist(gen);
  }
  return vec;
}

TEST(beresnev_a_increase_contrast_mpi, Incorrect_format) {
  boost::mpi::communicator world;
  double factor = 1.0;
  int width = 1;
  int height = 1;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> inp;

  std::string head =
      "BM\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inp = std::vector<uint8_t>(width * height * 3);

    input_buffer.reserve(file_size);
    input_buffer.insert(input_buffer.end(), head.begin(), head.end());
    input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

    out_buffer = std::vector<uint8_t>(file_size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }
  beresnev_a_increase_contrast_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  if (world.rank() == 0) ASSERT_EQ(testMPITaskParallel.validation(), false);
}

TEST(beresnev_a_increase_contrast_mpi, Incorrect_title) {
  boost::mpi::communicator world;
  double factor = 1.0;
  int width = 1;
  int height = 1;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> inp;

  std::string head =
      "P6\n" + std::to_string(width - 1) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inp = std::vector<uint8_t>(width * height * 3);

    input_buffer.reserve(file_size);
    input_buffer.insert(input_buffer.end(), head.begin(), head.end());
    input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

    out_buffer = std::vector<uint8_t>(file_size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }
  beresnev_a_increase_contrast_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  if (world.rank() == 0) ASSERT_EQ(testMPITaskParallel.validation(), false);
}

TEST(beresnev_a_increase_contrast_mpi, Incorrect_title_1) {
  boost::mpi::communicator world;
  double factor = 1.0;
  int width = 1;
  int height = 1;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> inp;

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color + 1) + '\n';

  file_size = head.size() + width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inp = std::vector<uint8_t>(width * height * 3);

    input_buffer.reserve(file_size);
    input_buffer.insert(input_buffer.end(), head.begin(), head.end());
    input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

    out_buffer = std::vector<uint8_t>(file_size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }
  beresnev_a_increase_contrast_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  if (world.rank() == 0) ASSERT_EQ(testMPITaskParallel.validation(), false);
}

TEST(beresnev_a_increase_contrast_mpi, Invalid_output_buffer) {
  boost::mpi::communicator world;
  double factor = 1.0;
  int width = 10;
  int height = 10;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> inp;

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inp = std::vector<uint8_t>(width * height * 3);

    input_buffer.reserve(file_size);
    input_buffer.insert(input_buffer.end(), head.begin(), head.end());
    input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

    out_buffer = std::vector<uint8_t>(file_size - 1);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size - 1);
  }
  beresnev_a_increase_contrast_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  if (world.rank() == 0) ASSERT_EQ(testMPITaskParallel.validation(), false);
}

TEST(beresnev_a_increase_contrast_mpi, Empty_image) {
  boost::mpi::communicator world;
  double factor = 1.8;
  int width = 0;
  int height = 0;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> inp;

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inp = getRandomVector(width * height * 3);

    input_buffer.reserve(file_size);
    input_buffer.insert(input_buffer.end(), head.begin(), head.end());
    input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

    out_buffer = std::vector<uint8_t>(file_size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }
  beresnev_a_increase_contrast_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> reference(file_size);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataSeq->inputs_count.emplace_back(file_size);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&reference));
    taskDataSeq->outputs_count.emplace_back(file_size);
    // Create Task
    beresnev_a_increase_contrast_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference, out_buffer);
  }
}

TEST(beresnev_a_increase_contrast_mpi, Empty_file) {
  boost::mpi::communicator world;
  double factor = 1.0;
  int width = 0;
  int height = 0;
  // int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> inp;

  std::string head;

  file_size = head.size() + width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inp = std::vector<uint8_t>(width * height * 3);

    input_buffer.reserve(file_size);
    input_buffer.insert(input_buffer.end(), head.begin(), head.end());
    input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

    out_buffer = std::vector<uint8_t>(file_size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }
  beresnev_a_increase_contrast_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  if (world.rank() == 0) ASSERT_EQ(testMPITaskParallel.validation(), false);
}

TEST(beresnev_a_increase_contrast_mpi, Test_Random) {
  boost::mpi::communicator world;
  double factor = 1.8;
  int width = 13;
  int height = 7;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> inp;

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inp = getRandomVector(width * height * 3);

    input_buffer.reserve(file_size);
    input_buffer.insert(input_buffer.end(), head.begin(), head.end());
    input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

    out_buffer = std::vector<uint8_t>(file_size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }
  beresnev_a_increase_contrast_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> reference(file_size);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataSeq->inputs_count.emplace_back(file_size);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&reference));
    taskDataSeq->outputs_count.emplace_back(file_size);
    // Create Task
    beresnev_a_increase_contrast_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference, out_buffer);
  }
}

TEST(beresnev_a_increase_contrast_mpi, Test_Random_1) {
  boost::mpi::communicator world;
  double factor = 1.3;
  int width = 1009;
  int height = 1019;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> inp;

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + width * height * 3;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    inp = getRandomVector(width * height * 3);

    input_buffer.reserve(file_size);
    input_buffer.insert(input_buffer.end(), head.begin(), head.end());
    input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

    out_buffer = std::vector<uint8_t>(file_size);

    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }
  beresnev_a_increase_contrast_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    // Create data
    std::vector<uint8_t> reference(file_size);

    // Create TaskData
    std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataSeq->inputs_count.emplace_back(file_size);
    taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataSeq->inputs_count.emplace_back(1);
    taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&reference));
    taskDataSeq->outputs_count.emplace_back(file_size);
    // Create Task
    beresnev_a_increase_contrast_mpi::TestMPITaskSequential testMpiTaskSequential(taskDataSeq);
    ASSERT_EQ(testMpiTaskSequential.validation(), true);
    testMpiTaskSequential.pre_processing();
    testMpiTaskSequential.run();
    testMpiTaskSequential.post_processing();
    ASSERT_EQ(reference, out_buffer);
  }
}

TEST(beresnev_a_increase_contrast_mpi, Test_file) {
  boost::mpi::communicator world;
  double factor = 2.5;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> ans_buffer;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::string input_file = "tasks/seq/beresnev_a_increase_contrast/input.ppm";
    std::string ans_file = "tasks/seq/beresnev_a_increase_contrast/ans.ppm";

    if (!std::filesystem::exists(input_file)) {
      input_file = "../../tasks/seq/beresnev_a_increase_contrast/input.ppm";
    }
    std::ifstream infile(input_file, std::ios::binary);
    ASSERT_EQ(!infile, false) << "Error: file not found in!" << std::endl;

    infile.seekg(0, std::ios::end);
    file_size = static_cast<size_t>(infile.tellg());
    infile.seekg(0, std::ios::beg);

    input_buffer = std::vector<uint8_t>(file_size);
    infile.read(reinterpret_cast<char *>(input_buffer.data()), file_size);
    ASSERT_EQ(!infile, false) << "Error reading file in!" << std::endl;

    infile.close();

    if (!std::filesystem::exists(ans_file)) {
      ans_file = "../../tasks/seq/beresnev_a_increase_contrast/ans.ppm";
    }
    std::ifstream ansfile(ans_file, std::ios::binary);
    ASSERT_EQ(!ansfile, false) << "Error: file not found ans!" << std::endl;

    ansfile.seekg(0, std::ios::end);
    ASSERT_EQ(file_size, static_cast<size_t>(ansfile.tellg())) << "Wrong input or answer" << std::endl;
    ansfile.seekg(0, std::ios::beg);

    ans_buffer = std::vector<uint8_t>(file_size);
    ansfile.read(reinterpret_cast<char *>(ans_buffer.data()), file_size);
    ASSERT_EQ(!ansfile, false) << "Error reading file ans!" << std::endl;

    ansfile.close();

    out_buffer = std::vector<uint8_t>(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }
  beresnev_a_increase_contrast_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(ans_buffer, out_buffer);
  }
}

TEST(beresnev_a_increase_contrast_mpi, Test_file_1) {
  boost::mpi::communicator world;
  double factor = 1.5;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> out_buffer;
  std::vector<uint8_t> ans_buffer;

  std::shared_ptr<ppc::core::TaskData> taskDataPar = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    std::string input_file = "tasks/seq/beresnev_a_increase_contrast/input1.ppm";
    std::string ans_file = "tasks/seq/beresnev_a_increase_contrast/ans1.ppm";

    if (!std::filesystem::exists(input_file)) {
      input_file = "../../tasks/seq/beresnev_a_increase_contrast/input1.ppm";
    }
    std::ifstream infile(input_file, std::ios::binary);
    ASSERT_EQ(!infile, false) << "Error: file not found in!" << std::endl;

    infile.seekg(0, std::ios::end);
    file_size = static_cast<size_t>(infile.tellg());
    infile.seekg(0, std::ios::beg);

    input_buffer = std::vector<uint8_t>(file_size);
    infile.read(reinterpret_cast<char *>(input_buffer.data()), file_size);
    ASSERT_EQ(!infile, false) << "Error reading file in!" << std::endl;

    infile.close();

    if (!std::filesystem::exists(ans_file)) {
      ans_file = "../../tasks/seq/beresnev_a_increase_contrast/ans1.ppm";
    }
    std::ifstream ansfile(ans_file, std::ios::binary);
    ASSERT_EQ(!ansfile, false) << "Error: file not found ans!" << std::endl;

    ansfile.seekg(0, std::ios::end);
    ASSERT_EQ(file_size, static_cast<size_t>(ansfile.tellg())) << "Wrong input or answer" << std::endl;
    ansfile.seekg(0, std::ios::beg);

    ans_buffer = std::vector<uint8_t>(file_size);
    ansfile.read(reinterpret_cast<char *>(ans_buffer.data()), file_size);
    ASSERT_EQ(!ansfile, false) << "Error reading file ans!" << std::endl;

    ansfile.close();

    out_buffer = std::vector<uint8_t>(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
    taskDataPar->inputs_count.emplace_back(file_size);
    taskDataPar->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
    taskDataPar->inputs_count.emplace_back(1);
    taskDataPar->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
    taskDataPar->outputs_count.emplace_back(file_size);
  }
  beresnev_a_increase_contrast_mpi::TestMPITaskParallel testMPITaskParallel(taskDataPar);
  ASSERT_EQ(testMPITaskParallel.validation(), true);
  testMPITaskParallel.pre_processing();
  testMPITaskParallel.run();
  testMPITaskParallel.post_processing();

  if (world.rank() == 0) {
    ASSERT_EQ(ans_buffer, out_buffer);
  }
}