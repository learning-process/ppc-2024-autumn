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

namespace voroshilov_v_torus_grid_mpi {

int select_path_proc(int current_id, int destination_id, int grid);
std::pair<int, int> select_terminate_proc(int current_id, int terminate_code, int grid);

struct Commands {
  static const int send_from_source = 0;
  static const int route_to_dest = 1;
  static const int move_to_zero = -100;
  static const int direct_terminate = 10;
  static const int reverse_terminate = -10;
};

struct Tags {
  static const int terminate_command = 1000;
  static const int current_proc = 0;
  static const int buf_size = 1;
  static const int buffer = 2;
  static const int path_size = 3;
  static const int path = 4;
};

class TorusGridTaskParallel : public ppc::core::Task {
 public:
  explicit TorusGridTaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  struct Message {
    std::vector<char> buffer;
    std::vector<int> path;
  } message;

  int grid_size;

  int source_proc;
  int destination_proc;
  int current_proc;

  int terminate_command;

  boost::mpi::communicator world;
};

}  // namespace voroshilov_v_torus_grid_mpi