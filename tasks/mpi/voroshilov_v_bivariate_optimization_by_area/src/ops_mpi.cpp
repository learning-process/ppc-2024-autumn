#include "mpi/voroshilov_v_bivariate_optimization_by_area/include/ops_mpi.hpp"

// Sequentional:

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskSequential::validation() {
  internal_order_test();

  // criterium-function length <= 0:
  if (taskData->inputs_count[0] <= 0) {
    return false;
  }
  // incorrect number of search areas:
  if (taskData->inputs_count[1] != 4) {
    return false;
  }
  // search areas:
  double* d_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  double x_min = *d_ptr++;
  double x_max = *d_ptr++;
  double y_min = *d_ptr++;
  double y_max = *d_ptr++;
  if (x_min > x_max) {
    return false;
  }
  if (y_min > y_max) {
    return false;
  }
  // incorrect number of steps count:
  if (taskData->inputs_count[2] != 2) {
    return false;
  }
  // steps_count x:
  if (taskData->inputs[2][0] <= 0) {
    return false;
  }
  // steps_count y:
  if (taskData->inputs[2][1] <= 0) {
    return false;
  }
  // constraints count is not equal as it is:
  int g_count = *reinterpret_cast<int*>(taskData->inputs[3]);
  if (g_count != (taskData->inputs).size() - 4) {
    return false;
  }

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskSequential::pre_processing() {
  internal_order_test();

  // criterium-function:
  std::vector<char> q_vec(taskData->inputs_count[0]);
  char* q_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(q_ptr, q_ptr + taskData->inputs_count[0], q_vec.begin());
  q = Polynomial(q_vec);

  // search area:
  double* d_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
  x_area.min_value = *d_ptr++;
  x_area.max_value = *d_ptr++;
  y_area.min_value = *d_ptr++;
  y_area.max_value = *d_ptr++;

  // steps counts:
  int* s_ptr = reinterpret_cast<int*>(taskData->inputs[2]);
  x_area.steps_count = *s_ptr++;
  y_area.steps_count = *s_ptr;

  int g_count = *reinterpret_cast<int*>(taskData->inputs[3]);

  // constraints-functions:
  for (int i = 4; i < 4 + g_count; i++) {
    char* g_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
    std::vector<char> current_g_vec(taskData->inputs_count[i]);
    std::copy(g_ptr, g_ptr + taskData->inputs_count[i], current_g_vec.begin());
    Polynomial current_g_pol(current_g_vec);
    g.push_back(current_g_pol);
  }

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskSequential::run() {
  internal_order_test();

  // Preparing vector of points:

  double x_step = (x_area.max_value - x_area.min_value) / x_area.steps_count;
  double y_step = (y_area.max_value - y_area.min_value) / y_area.steps_count;

  std::vector<Point> points;
  Point current_point(x_area.min_value, y_area.min_value);

  while (current_point.y <= y_area.max_value) {
    while (current_point.x <= x_area.max_value) {
      points.push_back(current_point);
      current_point.x += x_step;
    }
    current_point.x = x_area.min_value;
    current_point.y += y_step;
  }

  // Finding minimum in this vector of points:

  // Find first point satisfied constraints:
  int index = 0;
  bool flag_in_area = false;
  while ((index < points.size()) && (flag_in_area == false)) {
    // Check if constraints is satisfied:
    flag_in_area = true;
    for (int j = 0; j < g.size(); j++) {
      if (g[j].calculate(points[index]) > 0) {
        flag_in_area = false;
        break;
      }
    }
    index++;
  }
  int first_satisfied = index;  // it is first candidate for optimum
  optimum_point = points[first_satisfied];
  optimum_value = q.calculate(points[first_satisfied]);

  // Start search from this point:
  for (int i = first_satisfied + 1; i < points.size(); i++) {
    double current_value = q.calculate(points[i]);
    // Check if constraints is satisfied:
    flag_in_area = true;
    for (int j = 0; j < g.size(); j++) {
      if (g[j].calculate(points[i]) > 0) {
        flag_in_area = false;
        break;
      }
    }
    // Check if current < optimum
    if (flag_in_area == true) {
      if (current_value < optimum_value) {
        optimum_value = current_value;
        optimum_point.x = points[i].x;
        optimum_point.y = points[i].y;
      }
    }
  }

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = optimum_point.x;
  reinterpret_cast<double*>(taskData->outputs[0])[1] = optimum_point.y;
  reinterpret_cast<double*>(taskData->outputs[0])[2] = optimum_value;

  return true;
}

// Parallel:

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel::validation() {
  internal_order_test();

  if (world.rank() == 0) {
    // criterium-function length <= 0:
    if (taskData->inputs_count[0] <= 0) {
      return false;
    }
    // incorrect number of search areas:
    if (taskData->inputs_count[1] != 4) {
      return false;
    }
    // search areas:
    double* d_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
    double x_min = *d_ptr++;
    double x_max = *d_ptr++;
    double y_min = *d_ptr++;
    double y_max = *d_ptr++;
    if (x_min > x_max) {
      return false;
    }
    if (y_min > y_max) {
      return false;
    }
    // incorrect number of steps count:
    if (taskData->inputs_count[2] != 2) {
      return false;
    }
    // steps_count x:
    if (taskData->inputs[2][0] <= 0) {
      return false;
    }
    // steps_count y:
    if (taskData->inputs[2][1] <= 0) {
      return false;
    }
    // constraints count is not equal as it is:
    int g_count = *reinterpret_cast<int*>(taskData->inputs[3]);
    if (g_count != (taskData->inputs).size() - 4) {
      return false;
    }
  }
  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel::pre_processing() {
  internal_order_test();

  // criterium-function:
  std::vector<char> q_vec(taskData->inputs_count[0]);
  char* q_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(q_ptr, q_ptr + taskData->inputs_count[0], q_vec.begin());
  q = Polynomial(q_vec);

  if (world.rank() == 0) {
    // search area:
    double* d_ptr = reinterpret_cast<double*>(taskData->inputs[1]);
    double x_min = *d_ptr++;
    double x_max = *d_ptr++;
    double y_min = *d_ptr++;
    double y_max = *d_ptr++;

    // steps counts:
    int* s_ptr = reinterpret_cast<int*>(taskData->inputs[2]);
    int x_steps = *s_ptr++;
    int y_steps = *s_ptr;

    x_area = Search_area(x_min, x_max, x_steps);
    y_area = Search_area(y_min, y_max, y_steps);
  }

  int g_count = *reinterpret_cast<int*>(taskData->inputs[3]);

  // constraints-functions:
  for (int i = 4; i < 4 + g_count; i++) {
    char* g_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
    std::vector<char> current_g_vec(taskData->inputs_count[i]);
    std::copy(g_ptr, g_ptr + taskData->inputs_count[i], current_g_vec.begin());
    Polynomial current_g_pol(current_g_vec);
    g.push_back(current_g_pol);
  }

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_mpi::OptimizationMPITaskParallel::run() {
  internal_order_test();

  if (world.rank() == 0) {
    // Preparing vector of points:

    double x_step = (x_area.max_value - x_area.min_value) / x_area.steps_count;
    double y_step = (y_area.max_value - y_area.min_value) / y_area.steps_count;

    std::vector<Point> points;
    Point current_point(x_area.min_value, y_area.min_value);

    while (current_point.y <= y_area.max_value) {
      while (current_point.x <= x_area.max_value) {
        points.push_back(current_point);
        current_point.x += x_step;
      }
      current_point.x = x_area.min_value;
      current_point.y += y_step;
    }
  }

  // Finding minimum in this vector of points:

  // Find first point satisfied constraints:
  int index = 0;
  bool flag_in_area = false;
  while ((index < points.size()) && (flag_in_area == false)) {
    // Check if constraints is satisfied:
    flag_in_area = true;
    for (int j = 0; j < g.size(); j++) {
      if (g[j].calculate(points[index]) > 0) {
        flag_in_area = false;
        break;
      }
    }
    index++;
  }
  int first_satisfied = index;  // it is first candidate for optimum
  optimum_point = points[first_satisfied];
  optimum_value = q.calculate(points[first_satisfied]);

  // Start search from this point:
  for (int i = first_satisfied + 1; i < points.size(); i++) {
    double current_value = q.calculate(points[i]);
    // Check if constraints is satisfied:
    flag_in_area = true;
    for (int j = 0; j < g.size(); j++) {
      if (g[j].calculate(points[i]) > 0) {
        flag_in_area = false;
        break;
      }
    }
    // Check if current < optimum
    if (flag_in_area == true) {
      if (current_value < optimum_value) {
        optimum_value = current_value;
        optimum_point.x = points[i].x;
        optimum_point.y = points[i].y;
      }
    }
  }

  return true;
}

bool nesterov_a_test_task_mpi::TestMPITaskParallel::run() {
  internal_order_test();
  int local_res;
  if (ops == "+") {
    local_res = std::accumulate(local_input_.begin(), local_input_.end(), 0);
  } else if (ops == "-") {
    local_res = -std::accumulate(local_input_.begin(), local_input_.end(), 0);
  } else if (ops == "max") {
    local_res = *std::max_element(local_input_.begin(), local_input_.end());
  }

  if (ops == "+" || ops == "-") {
    reduce(world, local_res, res, std::plus(), 0);
  } else if (ops == "max") {
    reduce(world, local_res, res, boost::mpi::maximum<int>(), 0);
  }
  std::this_thread::sleep_for(20ms);
  return true;
}

bool nesterov_a_test_task_mpi::TestMPITaskParallel::post_processing() {
  internal_order_test();
  if (world.rank() == 0) {
    reinterpret_cast<int*>(taskData->outputs[0])[0] = res;
  }
  return true;
}
