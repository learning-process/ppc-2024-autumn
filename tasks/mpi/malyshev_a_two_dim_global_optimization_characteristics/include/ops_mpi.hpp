#pragma once
#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>
#include <functional>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace malyshev_a_two_dim_global_optimization_characteristics_mpi {

using constraint_t = std::function<bool(double, double)>;
using target_t = std::function<double(double, double)>;

struct Point {
  double x;
  double y;
  double value;

  Point(double x = 0, double y = 0) : x(x), y(y) { value = std::numeric_limits<double>::max(); }
  Point(double x, double y, double value) : x(x), y(y), value(value) {}

  template <typename Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar & x;
    ar & y;
    ar & value;
  }
};

class TestTaskSequential : public ppc::core::Task {
 public:
  explicit TestTaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_, target_t traget,
                              std::vector<constraint_t> constraints)
      : Task(std::move(taskData_)), traget_function_(std::move(traget)), constraints_(std::move(constraints)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 protected:
  struct Constants {
    static constexpr double h = 1e-7;
    static constexpr int max_iterations = 100;
    static constexpr int grid_initial_size = 20;
    static constexpr double tunnel_rate = 0.1;
    static constexpr double start_learning_rate = 0.01;
    static constexpr int num_tunnels = 20;
  };

  struct Data {
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double eps;

    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version) {
      ar & x_min;
      ar & x_max;
      ar & y_min;
      ar & y_max;
      ar & eps;
    }
  } data_;

  target_t traget_function_;
  std::vector<constraint_t> constraints_;
  Point res_;

  virtual Point local_search(double x0, double y0);
  virtual Point tunnel_search(const Point& current_min);
  virtual bool check_constraints(double x, double y);

  void readTaskData();
  void writeTaskData();
  bool validateTaskData();
  virtual void optimize();
};

class TestTaskParallel : public TestTaskSequential {
 public:
  explicit TestTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_, target_t traget,
                            std::vector<constraint_t> constraints)
      : TestTaskSequential(std::move(taskData_), std::move(traget), std::move(constraints)) {
    init_shared_bool_array();
  }
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

  ~TestTaskParallel() override { cleanup_shared_arrays(); }

 private:
  int start_constraint_index_;
  int local_constraint_count_;
  bool is_active_;

  MPI_Win bool_win;
  bool* shared_results;

  void distrebute_constraints();
  Point tunnel_search(const Point& current_min) override;
  bool check_constraints(double x, double y) override;
  Point local_search(double x0, double y0) override;

  void init_shared_bool_array();
  void cleanup_shared_arrays();

  boost::mpi::communicator world;
};

}  // namespace malyshev_a_two_dim_global_optimization_characteristics_mpi