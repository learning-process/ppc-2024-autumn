// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <filesystem>

#include "seq/beresnev_a_increase_contrast/include/ops_seq.hpp"

TEST(beresnev_a_increase_contrast_seq, Test_File) {
  double factor = 2.5;

  std::filesystem::path currentPath = std::filesystem::current_path();
  std::cerr << currentPath << std::endl;

  const std::string input_file = "tasks/seq/beresnev_a_increase_contrast/input.ppm";
  const std::string ans_file = "tasks/seq/beresnev_a_increase_contrast/o.ppm";

  std::ifstream infile(input_file, std::ios::binary);
  ASSERT_EQ(!infile, false) << "Error: file not found in!" << std::endl;

  infile.seekg(0, std::ios::end);
  size_t file_size = static_cast<size_t>(infile.tellg());
  infile.seekg(0, std::ios::beg);

  std::vector<uint8_t> input_buffer(file_size);
  infile.read(reinterpret_cast<char *>(input_buffer.data()), file_size);
  ASSERT_EQ(!infile, false) << "Error reading file in!" << std::endl;

  infile.close();

  std::ifstream ansfile(ans_file, std::ios::binary);
  ASSERT_EQ(!ansfile, false) << "Error: file not found ans!" << std::endl;

  ansfile.seekg(0, std::ios::end);
  ASSERT_EQ(file_size, static_cast<size_t>(ansfile.tellg())) << "Wrong input or answer" << std::endl;
  ansfile.seekg(0, std::ios::beg);

  std::vector<uint8_t> ans_buffer(file_size);
  ansfile.read(reinterpret_cast<char *>(ans_buffer.data()), file_size);
  ASSERT_EQ(!ansfile, false) << "Error reading file ans!" << std::endl;

  ansfile.close();

  std::vector<uint8_t> out_buffer(file_size);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
  taskDataSeq->inputs_count.emplace_back(file_size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
  taskDataSeq->outputs_count.emplace_back(file_size);

  beresnev_a_increase_contrast_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), true);
  testTaskSequential.pre_processing();
  testTaskSequential.run();
  testTaskSequential.post_processing();
  ASSERT_EQ(ans_buffer, out_buffer);
}
