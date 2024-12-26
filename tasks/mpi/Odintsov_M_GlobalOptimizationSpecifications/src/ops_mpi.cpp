
#include "mpi/Odintsov_M_GlobalOptimizationSpecifications/include/ops_mpi.hpp"

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::satisfies_constraints(double x, double y, int number_constraint,
                                                                            std::vector<double> constraint) {
  double check = constraint[number_constraint * 3] * x + constraint[number_constraint * 3 + 1] * y -
                 constraint[number_constraint * 3 + 2];
  return check <= 0;
}

double Odintsov_M_GlobalOptimizationSpecifications_mpi::calculate_function(double x, double y,
                                                                           std::vector<double> func) {
  return (x - func[0]) * (x - func[0]) + (y - func[1]) * (y - func[1]);
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential::validation() {
  internal_order_test();

  if (taskData->outputs_count[0] == 0) return false;

  if ((taskData->inputs_count[1] != 1) && (taskData->inputs_count[1] != 0)) return false;
  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential::pre_processing() {
  internal_order_test();

  step = *reinterpret_cast<double*>(taskData->inputs[3]);

  for (int i = 0; i < 4; i++) {
    area.push_back(reinterpret_cast<double*>(taskData->inputs[0])[i]);
  }

  count_constraint = taskData->inputs_count[0];
  constraint.resize(count_constraint * 3);
  ver = taskData->inputs_count[1];

  for (int i = 0; i < 2; i++) {
    funct.push_back(reinterpret_cast<double*>(taskData->inputs[1])[i]);
  }

  for (int i = 0; i < count_constraint * 3; i++) {
    constraint[i] = (reinterpret_cast<double*>(taskData->inputs[2])[i]);
  }

  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential::run() {
  internal_order_test();
  if (ver == 0)
    ans = 999999999999999;
  else
    ans = -999999999999999;

  double current_step = step;
  double tolerance = 1e-6;
  double previous_ans = std::numeric_limits<double>::max();
  int scale_factor = static_cast<int>(1.0 / current_step);

  while (current_step >= tolerance) {
    double local_minX = area[0];
    double local_minY = area[2];

    int int_minX = static_cast<int>(area[0] * scale_factor);
    int int_maxX = static_cast<int>(area[1] * scale_factor);
    int int_minY = static_cast<int>(area[2] * scale_factor);
    int int_maxY = static_cast<int>(area[3] * scale_factor);

    for (int x = int_minX; x < int_maxX; x++) {
      for (int y = int_minY; y < int_maxY; y++) {
        double real_x = x / static_cast<double>(scale_factor);
        double real_y = y / static_cast<double>(scale_factor);
        bool is_point_correct = true;
        for (int i = 0; i < count_constraint; i++) {
          is_point_correct = satisfies_constraints(real_x, real_y, i, constraint);

          if (!is_point_correct) {
            break;
          }
        }
        if (is_point_correct) {
          double value = calculate_function(real_x, real_y, funct);
          if (ver == 0) {
            if (value < ans) {
              ans = value;
              local_minX = real_x;
              local_minY = real_y;
            }
          } else if (ver == 1) {
            if (value > ans) {
              ans = value;
              local_minX = real_x;
              local_minY = real_y;
            }
          }
        }
      }
    }

    if (std::abs(previous_ans - ans) < tolerance) {
      break;
    }

    area[0] = std::max(local_minX - 2 * current_step, area[0]);
    area[1] = std::min(local_minX + 2 * current_step, area[1]);
    area[2] = std::max(local_minY - 2 * current_step, area[2]);
    area[3] = std::min(local_minY + 2 * current_step, area[3]);

    current_step /= 2.0;
    previous_ans = ans;
  }

  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPISequential::post_processing() {
  internal_order_test();
  reinterpret_cast<double*>(taskData->outputs[0])[0] = ans;
  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel::validation() {
  internal_order_test();
  if (com.rank() == 0) {
    if (taskData->outputs_count[0] == 0) return false;

    if ((taskData->inputs_count[1] != 1) && (taskData->inputs_count[1] != 0)) return false;
  }
  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel::pre_processing() {
  internal_order_test();
  if (com.rank() == 0) {
    step = *reinterpret_cast<double*>(taskData->inputs[3]);
    area.resize(4);
    for (int i = 0; i < 4; i++) {
      area[i] = (reinterpret_cast<double*>(taskData->inputs[0])[i]);
    }

    count_constraint = taskData->inputs_count[0];
    ver = taskData->inputs_count[1];

    for (int i = 0; i < 2; i++) {
      funct.push_back(reinterpret_cast<double*>(taskData->inputs[1])[i]);
    }
    for (int i = 0; i < count_constraint * 3; i++) {
      constraint.push_back(reinterpret_cast<double*>(taskData->inputs[2])[i]);
    }
  }
  loc_constr_size = 0;
  return true;
}

bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel::run() {
  internal_order_test();

  ans = (ver == 0) ? 999999999999999 : -999999999999999;

  if (com.rank() == 0) {
    loc_constr_size = std::max(1, (count_constraint + com.size() - 1) / com.size());
  }

  broadcast(com, count_constraint, 0);
  broadcast(com, loc_constr_size, 0);
  broadcast(com, step, 0);

  if (com.rank() != 0) {
    area.resize(4);
  }

  broadcast(com, area.data(), area.size(), 0);

  if (com.rank() == 0) {
    for (int pr = 1; pr < com.size(); pr++) {
      if (pr * loc_constr_size < count_constraint) {
        std::vector<double> send(3 * loc_constr_size, 0);
        for (int i = 0; i < 3 * loc_constr_size; i++) {
          send[i] = constraint[pr * loc_constr_size * 3 + i];
        }
        com.send(pr, 0, send.data(), send.size());
      }
    }

    for (int i = 0; i < 3 * loc_constr_size; i++) {
      local_constraint.push_back(constraint[i]);
    }
  } else if (com.rank() < count_constraint) {
    std::vector<double> buffer(loc_constr_size * 3, 0);
    com.recv(0, 0, buffer.data(), buffer.size());
    local_constraint.insert(local_constraint.end(), buffer.begin(), buffer.end());
  }

  double current_step = step;
  double tolerance = 1e-6;
  double previous_ans = std::numeric_limits<double>::max();
  std::vector<double> loc_area(4, 0);
  for (int i = 0; i < 4; i++) {
    loc_area[i] = area[i];
  }

  while (current_step >= tolerance) {
    double local_minX = loc_area[0];
    double local_minY = loc_area[2];

    int scale_factor = static_cast<int>(1.0 / current_step);
    int int_minX = static_cast<int>(loc_area[0] * scale_factor);
    int int_maxX = static_cast<int>(loc_area[1] * scale_factor);
    int int_minY = static_cast<int>(loc_area[2] * scale_factor);
    int int_maxY = static_cast<int>(loc_area[3] * scale_factor);

    for (int x = int_minX; x < int_maxX; x++) {
      for (int y = int_minY; y < int_maxY; y++) {
        double real_x = x / static_cast<double>(scale_factor);
        double real_y = y / static_cast<double>(scale_factor);

        int loc_flag = 1;
        int constr_sz = local_constraint.size() / 3;
        for (int i = 0; i < constr_sz; i++) {
          if (!satisfies_constraints(real_x, real_y, i, local_constraint)) {
            loc_flag = 0;
            break;
          }
        }

        gather(com, loc_flag, is_corret, 0);

        if (com.rank() == 0) {
          bool flag = true;
          int sz = is_corret.size();
          for (int i = 0; i < sz; i++) {
            if (is_corret[i] == 0) {
              flag = false;
              break;
            }
          }
          if (flag) {
            double value = calculate_function(real_x, real_y, funct);
            if (ver == 0) {
              if (value < ans) {
                ans = value;
                local_minX = real_x;
                local_minY = real_y;
              }
            } else if (ver == 1) {
              if (value > ans) {
                ans = value;
                local_minX = real_x;
                local_minY = real_y;
              }
            }
          }
        }
      }
    }

    if (com.rank() == 0) {
      if ((std::abs(previous_ans - ans) < tolerance)) {
        current_step = -1;
      }
      std::vector<double> new_area = loc_area;

      // Уточняем границы
      new_area[0] = std::max(local_minX - 2 * current_step, area[0]);
      new_area[1] = std::min(local_minX + 2 * current_step, area[1]);
      new_area[2] = std::max(local_minY - 2 * current_step, area[2]);
      new_area[3] = std::min(local_minY + 2 * current_step, area[3]);

      loc_area = new_area;
      previous_ans = ans;
    }

    broadcast(com, loc_area.data(), area.size(), 0);
    broadcast(com, current_step, 0);
  }

  return true;
}
bool Odintsov_M_GlobalOptimizationSpecifications_mpi::GlobalOptimizationSpecificationsMPIParallel::post_processing() {
  internal_order_test();
  if (com.rank() == 0) reinterpret_cast<double*>(taskData->outputs[0])[0] = ans;
  return true;
}