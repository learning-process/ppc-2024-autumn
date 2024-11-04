
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

namespace komshina_d_min_of_vector_elements_mpi {

std::vector<int> getRandomVector(int sz);

class MinOfVectorElementsTaskSequential : public ppc::core::Task {
 public:
  explicit MinOfVectorElementsTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_;
  int min_res{};
};

class MinOfVectorElementsTaskParallel : public ppc::core::Task {
 public:
  explicit MinOfVectorElementsTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> input_, local_input_;
  int min_res{};
  boost::mpi::communicator world;
};

}  // namespace komshina_d_min_of_vector_elements_mpi
