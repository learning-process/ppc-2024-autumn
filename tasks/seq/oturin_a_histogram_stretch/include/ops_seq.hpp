#pragma once

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace oturin_a_histogram_stretch_seq {

using errno_t = int;

#if defined(_WIN32) || defined(WIN32)
#else
errno_t fopen_s(FILE** f, const char* name, const char* mode);
#endif

const int BYTES_PER_PIXEL = 3;  // red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;

std::vector<uint8_t> ReadBMP(const char* filename, int& w, int& h);

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int width = 0;
  int height = 0;
  std::vector<uint8_t> input;
  std::vector<uint8_t> result;
};

}  // namespace oturin_a_histogram_stretch_seq
