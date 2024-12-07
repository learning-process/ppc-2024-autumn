#pragma once

#include <gtest/gtest.h>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "core/task/include/task.hpp"

namespace boost {
namespace serialization {

template <typename Archive>
void serialize(Archive &ar, std::vector<double> &v, const unsigned int version) {
    ar & v;  // Сериализуем весь вектор
}

}  // namespace serialization
}  // namespace boost

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi {

class MPIGaussianHorizontalSequential : public ppc::core::Task {
 public:
  explicit MPIGaussianHorizontalSequential(std::shared_ptr<ppc::core::TaskData> taskData_) 
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> matrix;
  std::vector<double> vector_b;
};

class MPIGaussianHorizontalParallel : public ppc::core::Task {
 public:
  explicit MPIGaussianHorizontalParallel(std::shared_ptr<ppc::core::TaskData> taskData_) 
      : Task(std::move(taskData_)) {}

  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> matrix;
  std::vector<double> vector_b;
  std::vector<double> local_matrix_part;
  std::vector<double> local_b_part;
  std::vector<double> res;
  boost::mpi::communicator world;
};

}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_mpi
