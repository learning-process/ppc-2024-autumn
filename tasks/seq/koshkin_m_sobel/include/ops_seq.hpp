#pragma once

#include <vector>

#include "core/task/include/task.hpp"

template <typename T>
void dumpv(const char* text, const std::vector<T>& v) {
  std::cout << "\n" << text << ": [";
  for (const auto& e : v) std::cout << (int)e << ", ";
  std::cout << "]\n";
}

namespace koshkin_m_sobel_seq {

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::pair<size_t, size_t> imgsize;
  std::vector<uint8_t> image;
  std::vector<uint8_t> resimg;

  // clang-format off
  static const std::array<std::array<int8_t, 3>, 3> SOBEL_KERNEL_X = {{
    {{-1, 0, 1}},
    {{-2, 0, 2}},
    {{-1, 0, 1}}
  }};
  static const std::array<std::array<int8_t, 3>, 3> SOBEL_KERNEL_Y = {{
    {{-1, -2, -1}},
    {{ 0,  0,  0}},
    {{ 1,  2,  1}}
  }};
  // clang-format on
};

}  // namespace koshkin_m_sobel_seq