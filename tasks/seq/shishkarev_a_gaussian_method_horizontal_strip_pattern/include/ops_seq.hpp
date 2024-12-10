#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace boost {
namespace serialization {

template <typename Archive>
void serialize(Archive &ar, std::vector<double> &v, const unsigned int version) {
  ar & v;  // Сериализуем весь вектор
}

}  // namespace serialization
}  // namespace boost

namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq {

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
}  // namespace shishkarev_a_gaussian_method_horizontal_strip_pattern_seq