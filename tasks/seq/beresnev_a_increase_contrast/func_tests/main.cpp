// Copyright 2023 Nesterov Alexander
#include <gtest/gtest.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "seq/beresnev_a_increase_contrast/include/ops_seq.hpp"

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

static std::vector<uint8_t> getAns(std::vector<uint8_t> &in, double f) {
  std::vector<uint8_t> out(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    double normalized = in[i] / 255.0;
    normalized = (normalized - 0.5) * f + 0.5;
    normalized = std::min(std::max(normalized, 0.0), 1.0);
    out[i] = static_cast<uint8_t>(normalized * 255);
  }
  return out;
}

TEST(beresnev_a_increase_contrast_seq, Incorrect_format) {
  double factor = 1.0;
  int width = 0;
  int height = 0;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> inp = getRandomVector(width * height * 3);

  std::string head =
      "P3\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + inp.size();

  input_buffer.reserve(file_size);
  input_buffer.insert(input_buffer.end(), head.begin(), head.end());
  input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

  std::vector<uint8_t> out_buffer(file_size);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
  taskDataSeq->inputs_count.emplace_back(file_size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
  taskDataSeq->outputs_count.emplace_back(file_size);

  beresnev_a_increase_contrast_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_increase_contrast_seq, Incorrect_title) {
  double factor = 1.0;
  int width = 10;
  int height = 10;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> inp = getRandomVector(width * height * 3);

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height + 1) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + inp.size();

  input_buffer.reserve(file_size);
  input_buffer.insert(input_buffer.end(), head.begin(), head.end());
  input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

  std::vector<uint8_t> ans_buffer;
  std::vector<uint8_t> ans = getAns(inp, factor);

  ans_buffer.reserve(file_size);
  ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
  ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

  std::vector<uint8_t> out_buffer(file_size);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
  taskDataSeq->inputs_count.emplace_back(file_size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
  taskDataSeq->outputs_count.emplace_back(file_size);

  beresnev_a_increase_contrast_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_increase_contrast_seq, Incorrect_title_1) {
  double factor = 1.0;
  int width = 10;
  int height = 10;
  int max_color = 253;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> inp = getRandomVector(width * height * 3);

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + inp.size();

  input_buffer.reserve(file_size);
  input_buffer.insert(input_buffer.end(), head.begin(), head.end());
  input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

  std::vector<uint8_t> ans_buffer;
  std::vector<uint8_t> ans = getAns(inp, factor);

  ans_buffer.reserve(file_size);
  ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
  ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

  std::vector<uint8_t> out_buffer(file_size);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
  taskDataSeq->inputs_count.emplace_back(file_size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
  taskDataSeq->outputs_count.emplace_back(file_size);

  beresnev_a_increase_contrast_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_increase_contrast_seq, Invalid_output_buffer) {
  double factor = 1.0;
  int width = 10;
  int height = 10;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> inp = getRandomVector(width * height * 3);

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + inp.size();

  input_buffer.reserve(file_size);
  input_buffer.insert(input_buffer.end(), head.begin(), head.end());
  input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

  std::vector<uint8_t> ans_buffer;
  std::vector<uint8_t> ans = getAns(inp, factor);

  ans_buffer.reserve(file_size);
  ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
  ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

  std::vector<uint8_t> out_buffer(file_size + 1);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
  taskDataSeq->inputs_count.emplace_back(file_size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
  taskDataSeq->outputs_count.emplace_back(file_size + 1);

  beresnev_a_increase_contrast_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_increase_contrast_seq, Empty_image) {
  double factor = 1.2;
  int width = 0;
  int height = 0;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> inp = getRandomVector(width * height * 3);

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + inp.size();

  input_buffer.reserve(file_size);
  input_buffer.insert(input_buffer.end(), head.begin(), head.end());
  input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

  std::vector<uint8_t> ans_buffer;
  std::vector<uint8_t> ans = getAns(inp, factor);

  ans_buffer.reserve(file_size);
  ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
  ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

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

TEST(beresnev_a_increase_contrast_seq, Empty_file) {
  double factor = 1.3;
  int width = 0;
  int height = 0;
  // int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> inp = getRandomVector(width * height * 3);

  std::string head;
  file_size = head.size() + inp.size();

  input_buffer.reserve(file_size);
  input_buffer.insert(input_buffer.end(), head.begin(), head.end());
  input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

  std::vector<uint8_t> ans_buffer;
  std::vector<uint8_t> ans = getAns(inp, factor);

  ans_buffer.reserve(file_size);
  ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
  ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

  std::vector<uint8_t> out_buffer(file_size);
  std::shared_ptr<ppc::core::TaskData> taskDataSeq = std::make_shared<ppc::core::TaskData>();
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input_buffer.data()));
  taskDataSeq->inputs_count.emplace_back(file_size);
  taskDataSeq->inputs.emplace_back(reinterpret_cast<uint8_t *>(&factor));
  taskDataSeq->inputs_count.emplace_back(1);
  taskDataSeq->outputs.emplace_back(reinterpret_cast<uint8_t *>(&out_buffer));
  taskDataSeq->outputs_count.emplace_back(file_size);

  beresnev_a_increase_contrast_seq::TestTaskSequential testTaskSequential(taskDataSeq);
  ASSERT_EQ(testTaskSequential.validation(), false);
}

TEST(beresnev_a_increase_contrast_seq, Test_Random) {
  double factor = 1.8;
  int width = 13;
  int height = 7;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> inp = getRandomVector(width * height * 3);

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + inp.size();

  input_buffer.reserve(file_size);
  input_buffer.insert(input_buffer.end(), head.begin(), head.end());
  input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

  std::vector<uint8_t> ans_buffer;
  std::vector<uint8_t> ans = getAns(inp, factor);

  ans_buffer.reserve(file_size);
  ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
  ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

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

TEST(beresnev_a_increase_contrast_seq, Test_Random_1) {
  double factor = 1.361;
  int width = 997;
  int height = 1039;
  int max_color = 255;

  size_t file_size;

  std::vector<uint8_t> input_buffer;
  std::vector<uint8_t> inp = getRandomVector(width * height * 3);

  std::string head =
      "P6\n" + std::to_string(width) + " " + std::to_string(height) + '\n' + std::to_string(max_color) + '\n';

  file_size = head.size() + inp.size();

  input_buffer.reserve(file_size);
  input_buffer.insert(input_buffer.end(), head.begin(), head.end());
  input_buffer.insert(input_buffer.end(), inp.data(), inp.data() + inp.size());

  std::vector<uint8_t> ans_buffer;
  std::vector<uint8_t> ans = getAns(inp, factor);

  ans_buffer.reserve(file_size);
  ans_buffer.insert(ans_buffer.end(), head.begin(), head.end());
  ans_buffer.insert(ans_buffer.end(), ans.data(), ans.data() + ans.size());

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

TEST(beresnev_a_increase_contrast_seq, Test_File) {
  double factor = 2.5;

  std::string input_file = "tasks/seq/beresnev_a_increase_contrast/input.ppm";
  std::string ans_file = "tasks/seq/beresnev_a_increase_contrast/ans.ppm";

  if (!std::filesystem::exists(input_file)) {
    input_file = "../../tasks/seq/beresnev_a_increase_contrast/input.ppm";
  }
  std::ifstream infile(input_file, std::ios::binary);
  ASSERT_EQ(!infile, false) << "Error: file not found in!" << std::endl;

  infile.seekg(0, std::ios::end);
  size_t file_size = static_cast<size_t>(infile.tellg());
  infile.seekg(0, std::ios::beg);

  std::vector<uint8_t> input_buffer(file_size);
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

TEST(beresnev_a_increase_contrast_seq, Test_File_1) {
  double factor = 1.5;

  std::string input_file = "tasks/seq/beresnev_a_increase_contrast/input1.ppm";
  std::string ans_file = "tasks/seq/beresnev_a_increase_contrast/ans1.ppm";

  if (!std::filesystem::exists(input_file)) {
    input_file = "../../tasks/seq/beresnev_a_increase_contrast/input1.ppm";
  }
  std::ifstream infile(input_file, std::ios::binary);
  ASSERT_EQ(!infile, false) << "Error: file not found in!" << std::endl;

  infile.seekg(0, std::ios::end);
  size_t file_size = static_cast<size_t>(infile.tellg());
  infile.seekg(0, std::ios::beg);

  std::vector<uint8_t> input_buffer(file_size);
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