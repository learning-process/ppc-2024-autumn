#include "mpi/voroshilov_v_torus_grid/include/ops_mpi.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <thread>
#include <vector>

int voroshilov_v_torus_grid_mpi::select_path_proc(int current_id, int destination_id, int grid) {
  int destination_row_id = destination_id / grid;
  int destination_col_id = destination_id % grid;
  int current_row_id = current_id / grid;
  int current_col_id = current_id % grid;

  int next_row_id = current_row_id;
  int next_col_id = current_col_id;

  if (destination_row_id - current_row_id != 0) {
    // Row is not the same

    // Destination is lower
    if (destination_row_id - current_row_id > 0) {
      if (destination_row_id - current_row_id <= grid / 2) {
        // Direct way
        next_row_id = current_row_id + 1;
        if (next_row_id == grid) {
          next_row_id = 0;
        }
      }
      if (destination_row_id - current_row_id > grid / 2) {
        // Reverse way
        next_row_id = current_row_id - 1;
        if (next_row_id == -1) {
          next_row_id = grid - 1;
        }
      }
    }

    // Destination is higher
    if (destination_row_id - current_row_id < 0) {
      if (-1 * (destination_row_id - current_row_id) <= grid / 2) {
        // Direct way
        next_row_id = current_row_id - 1;
        if (next_row_id == -1) {
          next_row_id = grid - 1;
        }
      }
      if (-1 * (destination_row_id - current_row_id) > grid / 2) {
        // Reverse way
        next_row_id = current_row_id + 1;
        if (next_row_id == grid) {
          next_row_id = 0;
        }
      }
    }

  } else {
    // Row is the same

    // Destination is on right
    if (destination_col_id - current_col_id > 0) {
      if (destination_col_id - current_col_id <= grid / 2) {
        // Direct way
        next_col_id = current_col_id + 1;
        if (next_col_id == grid) {
          next_col_id = 0;
        }
      }
      if (destination_col_id - current_col_id > grid / 2) {
        // Reverse way
        next_col_id = current_col_id - 1;
        if (next_col_id == -1) {
          next_col_id = grid - 1;
        }
      }
    }

    // Destination is on left
    if (destination_col_id - current_col_id < 0) {
      if (-1 * (destination_col_id - current_col_id) <= grid / 2) {
        // Direct way
        next_col_id = current_col_id - 1;
        if (next_col_id == -1) {
          next_col_id = grid - 1;
        }
      }
      if (-1 * (destination_col_id - current_col_id) > grid / 2) {
        // Reverse way
        next_col_id = current_col_id + 1;
        if (next_col_id == grid) {
          next_col_id = 0;
        }
      }
    }
  }

  int next_id = next_row_id * grid + next_col_id;

  return next_id;
}

std::pair<int, int> voroshilov_v_torus_grid_mpi::select_terminate_pair(int current_id, int grid) {
  int current_row_id = current_id / grid;
  int current_col_id = current_id % grid;

  int right_row_id = -1;
  int right_col_id = -1;
  int lower_row_id = -1;
  int lower_col_id = -1;

  if ((current_row_id == 0) && (current_col_id < grid - 1)) {
    // Route command do_terminate to right in first row
    right_row_id = current_row_id;
    right_col_id = current_col_id + 1;
  }
  if (current_row_id < grid - 1) {
    // Not last row, route command do_terminate to lower in column
    lower_row_id = current_row_id + 1;
    lower_col_id = current_col_id;
  }

  int right_id;
  if ((right_row_id == -1) || (right_col_id == -1)) {
    right_id = -1;
  } else {
    right_id = right_row_id * grid + right_col_id;
  }
  int lower_id;
  if ((lower_row_id == -1) || (lower_col_id == -1)) {
    lower_id = -1;
  } else {
    lower_id = lower_row_id * grid + lower_col_id;
  }

  std::pair<int, int> terminate_pair(right_id, lower_id);
  return terminate_pair;
}

bool voroshilov_v_torus_grid_mpi::TorusGridTaskParallel::validation() {
  internal_order_test();
  int world_size = world.size();

  int src = taskData->inputs_count[0];
  int dst = taskData->inputs_count[1];

  if ((src >= world_size) || (dst >= world_size)) {
    return false;
  }

  // Check if there is n^2 processes to build grid
  int n = sqrt(world_size);
  if (n * n != world_size) {
    return false;
  }

  if (world.rank() == src) {
    if (taskData->inputs_count[2] <= 0) {
      return false;
    };
  }

  if (world.rank() == dst) {
    if (taskData->outputs_count[0] <= 0) {
      return false;
    }
  }
  return true;
}

bool voroshilov_v_torus_grid_mpi::TorusGridTaskParallel::pre_processing() {
  internal_order_test();

  source_proc = taskData->inputs_count[0];
  destination_proc = taskData->inputs_count[1];

  if ((world.rank() == source_proc) || (world.rank() == destination_proc)) {
    buffer = std::vector<char>(taskData->inputs_count[2]);
    path = std::vector<int>{};
    auto* ptr = reinterpret_cast<char*>(taskData->inputs[0]);
    std::copy(ptr, ptr + taskData->inputs_count[2], buffer.begin());
  }
  int world_size = world.size();
  grid_size = sqrt(world_size);

  commands = {0, 1, 2, 3};

  tags = {0, 1, 2, 3, 4, 5};

  return true;
}

bool voroshilov_v_torus_grid_mpi::TorusGridTaskParallel::run() {
  internal_order_test();

  if (source_proc == destination_proc) {
    path.push_back(source_proc);
    return true;
  }

  if (world.rank() == source_proc) {
    terminate_command = commands.send_from_source;
    current_proc = source_proc;
  } else {
    world.recv(boost::mpi::any_source, tags.terminate_command, terminate_command);
    world.recv(boost::mpi::any_source, tags.current_proc, current_proc);
  }

  if (terminate_command == commands.route_to_dest) {
    // sending data is in progress and we are not source process

    size_t buf_size;
    world.recv(boost::mpi::any_source, tags.buf_size, buf_size);

    buffer = std::vector<char>(buf_size);
    world.recv(boost::mpi::any_source, tags.buffer, buffer.data(), buf_size);

    size_t path_size;
    world.recv(boost::mpi::any_source, tags.path_size, path_size);

    path = std::vector<int>(path_size);
    world.recv(boost::mpi::any_source, tags.path, path.data(), path_size);
  }

  if (terminate_command == commands.send_from_source || terminate_command == commands.route_to_dest) {
    // we are source process || we are router (or destination process)

    if (current_proc == destination_proc) {
      path.push_back(current_proc);
      terminate_command = commands.move_to_zero;
    } else {
      int next_proc = select_path_proc(current_proc, destination_proc, grid_size);
      path.push_back(current_proc);
      world.send(next_proc, tags.terminate_command, commands.route_to_dest);
      world.send(next_proc, tags.current_proc, next_proc);

      world.send(next_proc, tags.buf_size, buffer.size());
      world.send(next_proc, tags.buffer, buffer.data(), buffer.size());
      world.send(next_proc, tags.path_size, path.size());
      world.send(next_proc, tags.path, path.data(), path.size());

      world.recv(boost::mpi::any_source, tags.terminate_command, terminate_command);
      world.recv(boost::mpi::any_source, tags.current_proc, current_proc);
    }
  }

  if (terminate_command == commands.move_to_zero) {
    // sending data is completed and we are moving to process 0
    // in order to start terminating from him but now we are not terminating

    int next_proc = select_path_proc(current_proc, 0, grid_size);
    if (current_proc != next_proc) {
      if (next_proc == 0) {
        world.send(next_proc, tags.terminate_command, commands.do_terminate);
        world.send(next_proc, tags.current_proc, next_proc);
      } else {
        world.send(next_proc, tags.terminate_command, commands.move_to_zero);
        world.send(next_proc, tags.current_proc, next_proc);
      }
      world.recv(boost::mpi::any_source, tags.terminate_command, terminate_command);
      world.recv(boost::mpi::any_source, tags.current_proc, current_proc);
    } else {
      terminate_command = commands.do_terminate;
    }
  }

  if (terminate_command == commands.do_terminate) {
    // continue terminating processes

    std::pair<int, int> terminate_pair = select_terminate_pair(current_proc, grid_size);

    int right_proc = terminate_pair.first;
    if (right_proc > -1) {
      // If its first row and not last column send do_terminate command to right
      world.send(right_proc, tags.terminate_command, commands.do_terminate);
      world.send(right_proc, tags.current_proc, right_proc);
    }

    int lower_proc = terminate_pair.second;
    if (lower_proc > -1) {
      // If its not last row send do_terminate command to lower
      world.send(lower_proc, tags.terminate_command, commands.do_terminate);
      world.send(lower_proc, tags.current_proc, lower_proc);
    }
  }
  return true;
}

bool voroshilov_v_torus_grid_mpi::TorusGridTaskParallel::post_processing() {
  internal_order_test();

  if (world.rank() == destination_proc) {
    auto* ptr1 = reinterpret_cast<char*>(taskData->outputs[0]);
    std::copy(buffer.begin(), buffer.end(), ptr1);

    auto* ptr2 = reinterpret_cast<int*>(taskData->outputs[1]);
    std::copy(path.begin(), path.end(), ptr2);
  }

  // Without barrier() it works incorrect on large number of processes
  world.barrier();

  return true;
}
