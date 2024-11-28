// Filateva Elizaveta Metod Gausa
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace filateva_e_metod_gausa_mpi {


class MetodGausa : public ppc::core::Task {
 public:
  explicit MetodGausa(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int size;
  std::vector<double> matrix;
  std::vector<double> b_vector;
  std::vector<double> resh;
  boost::mpi::communicator world;
  boost::mpi::status status;
};

}  // namespace filateva_e_metod_gausa_mpi