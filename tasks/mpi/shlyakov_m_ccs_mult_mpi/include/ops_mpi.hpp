// Copyright 2023 Nesterov Alexander
#pragma once

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>
#include <memory>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "core/task/include/task.hpp"

namespace shlyakov_m_ccs_mult_mpi {

struct SparseMatrix {
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_pointers;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & values;
    ar & row_indices;
    ar & col_pointers;
  }
};

class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {
    std::srand(std::time(nullptr));
  }
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  SparseMatrix A_;
  int rows_a;
  int cols_a;
  SparseMatrix B_;
  int rows_b;
  int cols_b;
  SparseMatrix result_;

  boost::mpi::communicator world;
};

}  // namespace shlyakov_m_ccs_mult_mpi
