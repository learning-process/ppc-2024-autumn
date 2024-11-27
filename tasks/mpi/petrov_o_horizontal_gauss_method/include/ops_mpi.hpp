#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/serialization.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace petrov_o_horizontal_gauss_method_mpi { // Изменено название неймспейса

class SequentialTask : public ppc::core::Task {
 public:
  explicit SequentialTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<std::vector<double>> matrix;
  std::vector<double> b;
  std::vector<double> x;
};

class ParallelTask : public ppc::core::Task {
 public:
  explicit ParallelTask(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
	//size_t n;
	// std::vector<std::vector<double>> matrix;
	// std::vector<double> b;
	// std::vector<double> x;
	// std::vector<double> augmented_matrix;
	// double current_b;
	boost::mpi::communicator world;
	std::vector<std::vector<double>> matrix;
	std::vector<double> b, x;

  // size_t n;
  // int rank;
  // int size;


};

}  // namespace petrov_o_horizontal_gauss_method_mpi