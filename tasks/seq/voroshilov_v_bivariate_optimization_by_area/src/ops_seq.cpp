#include "seq/voroshilov_v_bivariate_optimization_by_area/include/ops_seq.hpp"

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::validation() {
  internal_order_test();

  // criterium-function length <= 0:
  if (taskData->inputs_count[0] <= 0) {
    return false;
  }
  // constraints count is not equal as it is:
  size_t g_count = *reinterpret_cast<size_t*>(taskData->inputs[1]);
  if (g_count != (taskData->inputs).size() - 4) {
    return false;
  }
  // incorrect number of search areas:
  if (taskData->inputs_count[2 + g_count] != 4) {
    return false;
  }
  // search areas:
  auto* d_ptr = reinterpret_cast<double*>(taskData->inputs[2 + g_count]);
  double x_min = *d_ptr++;
  double x_max = *d_ptr++;
  double y_min = *d_ptr++;
  double y_max = *d_ptr;
  if ((x_min > x_max) || (y_min > y_max)) {
    return false;
  }
  // incorrect number of steps count:
  if (taskData->inputs_count[2 + g_count + 1] != 2) {
    return false;
  }
  // steps counts:
  auto* s_ptr = reinterpret_cast<int*>(taskData->inputs[2 + g_count + 1]);
  int x_steps = *s_ptr++;
  int y_steps = *s_ptr;
  return (x_steps <= 0) && (y_steps <= 0);
}

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::pre_processing() {
  internal_order_test();

  // criterium-function:
  std::vector<char> q_vec(taskData->inputs_count[0]);
  char* q_ptr = reinterpret_cast<char*>(taskData->inputs[0]);
  std::copy(q_ptr, q_ptr + taskData->inputs_count[0], q_vec.begin());
  q = Polynomial(q_vec);

  size_t g_count = *reinterpret_cast<int*>(taskData->inputs[1]);

  // constraints-functions:
  for (size_t i = 2; i < 2 + g_count; i++) {
    char* g_ptr = reinterpret_cast<char*>(taskData->inputs[i]);
    std::vector<char> current_g_vec(taskData->inputs_count[i]);
    std::copy(g_ptr, g_ptr + taskData->inputs_count[i], current_g_vec.begin());
    Polynomial current_g_pol(current_g_vec);
    g.push_back(current_g_pol);
  }

  // search area:
  auto* d_ptr = reinterpret_cast<double*>(taskData->inputs[2 + g_count]);
  double x_min = *d_ptr++;
  double x_max = *d_ptr++;
  double y_min = *d_ptr++;
  double y_max = *d_ptr++;

  // steps counts:
  int* s_ptr = reinterpret_cast<int*>(taskData->inputs[2 + g_count + 1]);
  int x_steps = *s_ptr++;
  int y_steps = *s_ptr;

  x_area = Search_area(x_min, x_max, x_steps);
  y_area = Search_area(y_min, y_max, y_steps);

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::run() {
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
  size_t index = 0;
  bool flag_in_area = false;
  while ((index < points.size()) && (!flag_in_area)) {
    // Check if constraints is satisfied:
    flag_in_area = true;
    for (size_t j = 0; j < g.size(); j++) {
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
  for (size_t i = first_satisfied + 1; i < points.size(); i++) {
    double current_value = q.calculate(points[i]);
    // Check if constraints is satisfied:
    flag_in_area = true;
    for (size_t j = 0; j < g.size(); j++) {
      if (g[j].calculate(points[i]) > 0) {
        flag_in_area = false;
        break;
      }
    }
    // Check if current < optimum
    if (flag_in_area) {
      if (current_value < optimum_value) {
        optimum_value = current_value;
        optimum_point.x = points[i].x;
        optimum_point.y = points[i].y;
      }
    }
  }

  return true;
}

bool voroshilov_v_bivariate_optimization_by_area_seq::OptimizationTaskSequential::post_processing() {
  internal_order_test();

  reinterpret_cast<double*>(taskData->outputs[0])[0] = optimum_point.x;
  reinterpret_cast<double*>(taskData->outputs[0])[1] = optimum_point.y;
  reinterpret_cast<double*>(taskData->outputs[0])[2] = optimum_value;

  return true;
}
