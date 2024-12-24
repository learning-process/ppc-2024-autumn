#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

template <typename T>
void dumpv(const char* text, const std::vector<T>& v) {
  std::cout << "\n" << text << ": [";
  for (const auto& e : v) std::cout << (int)e << ", ";
  std::cout << "]\n";
}

namespace koshkin_m_sobel_mpi {

// clang-format off
static constexpr std::array<std::array<int8_t, 3>, 3> SOBEL_KERNEL_X = {{
  {{-1, 0, 1}},
  {{-2, 0, 2}},
  {{-1, 0, 1}}
}};
static constexpr std::array<std::array<int8_t, 3>, 3> SOBEL_KERNEL_Y = {{
  {{-1, -2, -1}},
  {{ 0,  0,  0}},
  {{ 1,  2,  1}}
}};
// clang-format on

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
};

class TestTaskParallel : public ppc::core::Task {
 public:
  explicit TestTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::pair<size_t, size_t> imgsize;
  std::vector<uint8_t> image;
  std::vector<uint8_t> resimg;

  boost::mpi::communicator world;
};

}  // namespace koshkin_m_sobel_mpi