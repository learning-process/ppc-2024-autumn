#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace leontev_n_image_enhancement_mpi {

class MPIImgEnhancementSequential : public ppc::core::Task {
 public:
  explicit MPIImgEnhancementSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> I;
  std::vector<int> image_input;
  std::vector<int> image_output;
};

class MPIImgEnhancementParallel : public ppc::core::Task {
 public:
  explicit MPIImgEnhancementParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> image_input;
  std::vector<int> image_output;
  boost::mpi::communicator world;
};
}  // namespace leontev_n_image_enhancement_mpi