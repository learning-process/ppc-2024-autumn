#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace somov_i_horizontal_scheme {

class MatrixVectorTask : public ppc::core::Task {
 public:
  explicit MatrixVectorTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  int32_t getRowCount() const { return rowCount_; }
  void setRowCount(int32_t rowCount) { rowCount_ = rowCount; }

  int32_t getColCount() const { return colCount_; }
  void setColCount(int32_t colCount) { colCount_ = colCount; }

 private:
  std::vector<int32_t> matrix_;
  std::vector<int32_t> vector_;
  std::vector<int32_t> result_;

  int32_t rowCount_;
  int32_t colCount_;
};

void distribute_matrix_rows(int32_t row, int32_t col, int32_t numProc, std::vector<int32_t>& matrix_sizes,
                            std::vector<int32_t>& peremeshcheniye_s);

class MatrixVectorTaskMPI : public ppc::core::Task {
 public:
  explicit MatrixVectorTaskMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int32_t> matrix_;
  std::vector<int32_t> vector_;
  std::vector<int32_t> result_;

  int32_t rowCount_;
  int32_t colCount_;

  std::vector<int32_t> raspredeleniye;
  std::vector<int32_t> peremeshcheniye;

  boost::mpi::communicator world;
};

}  // namespace somov_i_horizontal_scheme
