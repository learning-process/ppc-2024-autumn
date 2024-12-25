#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace smirnov_i_binary_segmentation {

class TestMPITaskSequential : public ppc::core::Task {
 public:
  explicit TestMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int cols;
  int rows;
  std::vector<int> img;
  std::vector<int> mask;
  static void merge_equivalence(std::map<int, std::set<int>>& eq_table, int a, int b);
  static std::vector<int> make_border(const std::vector<int>& img_, int cols_, int rows_);
  static std::vector<int> del_border(const std::vector<int>& img_, int cols_, int rows_);
};

class TestMPITaskParallel : public ppc::core::Task {
 public:
  explicit TestMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  int cols;
  int rows;
  std::vector<int> img;
  std::vector<int> mask;
  static void merge_equivalence(std::map<int, std::set<int>>& eq_table, int a, int b);
  static std::vector<int> make_border(const std::vector<int>& img_, int cols_, int rows_);
  static std::vector<int> del_border(const std::vector<int>& img_, int cols_, int rows_);
  static std::string serialize_eq_table(const std::map<int, std::set<int>>& eq_table);
  static std::map<int, std::set<int>> deserialize_eq_table(const std::string& str);
  static void merge_global_eq_table(std::map<int, std::set<int>>& global_eq_table,
                                    const std::map<int, std::set<int>>& local_eq_table);
  boost::mpi::communicator world;
};

}  // namespace smirnov_i_binary_segmentation