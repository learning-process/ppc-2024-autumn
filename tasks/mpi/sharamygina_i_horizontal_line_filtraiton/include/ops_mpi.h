#pragma once
#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"

namespace sharamygina_i_horizontal_line_filtration_mpi {
class horizontal_line_filtration_mpi : public ppc::core::Task {
 public:
  explicit horizontal_line_filtration_mpi(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  unsigned int gauss[3][3]{{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
  std::vector<unsigned int> original_data_;
  std::vector<unsigned int> result_data_;
  int rows_;
  int cols_;
  unsigned int InputAnotherPixel(const std::vector<unsigned int>& image, int x, int y, int rows, int cols);
  boost::mpi::communicator world;
}
}  // namespace sharamygina_i_horizontal_line_filtration_mpi