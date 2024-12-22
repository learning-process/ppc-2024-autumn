// Filatev Vladislav Metod Belmana Forda
#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <vector>

#include "core/task/include/task.hpp"

namespace filatev_v_metod_belmana_forda_mpi {

class MetodBelmanaFordaMPI : public ppc::core::Task {
 public:
  explicit MetodBelmanaFordaMPI(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  boost::mpi::communicator world;
  int n;
  int m;
  int start;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;
};

class MetodBelmanaFordaSeq : public ppc::core::Task {
 public:
  explicit MetodBelmanaFordaSeq(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {};
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int n;
  int m;
  int start;
  std::vector<int> Adjncy;
  std::vector<int> Xadj;
  std::vector<int> Eweights;
  std::vector<int> d;
};

}  // namespace filatev_v_metod_belmana_forda_mpi