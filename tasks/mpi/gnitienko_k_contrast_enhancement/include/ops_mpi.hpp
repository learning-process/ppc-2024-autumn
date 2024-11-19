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

namespace gnitienko_k_contrast_enhancement_mpi {

std::vector<int> getRandomVector(int sz);

class ContrastEnhanceSeq : public ppc::core::Task {
 public:
  explicit ContrastEnhanceSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> image;
  std::vector<int> res;
  double contrast_factor{};
  bool is_grayscale() const;
};

class ContrastEnhanceMPI : public ppc::core::Task {
 public:
  explicit ContrastEnhanceMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> image;
  size_t img_size{};
  std::vector<int> res;
  double contrast_factor{};
  bool is_grayscale() const;
  boost::mpi::communicator world;
};

}  // namespace gnitienko_k_contrast_enhancement_mpi