#pragma once

#include <boost/mpi.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace milovankin_m_component_labeling_mpi {

//
// Sequential
//
class ComponentLabelingSeq : public ppc::core::Task {
 public:
  explicit ComponentLabelingSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<uint8_t> input_image_;
  std::vector<uint32_t> labels_;
  size_t rows;
  size_t cols;
};

//
// Parallel
//
class ComponentLabelingPar : public ppc::core::Task {
 public:
  explicit ComponentLabelingPar(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;

  std::vector<uint8_t> input_image_;
  std::vector<uint8_t> local_image_;
  std::vector<uint32_t> labels_;
  size_t rows;
  size_t cols;
};
}  // namespace milovankin_m_component_labeling_mpi