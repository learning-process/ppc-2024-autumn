//Golovkin Maksims

#pragma once

#include <memory>
#include <vector>
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace golovkin_integration_rectangular_method {
class IntegralCalculator : public ppc::core::Task { 
 public:
  explicit IntegralCalculator(std::shared_ptr<ppc::core::TaskData> taskData);

  bool validation();       
  bool pre_processing(); 
  bool post_processing();
  bool run();              

 private:
  std::shared_ptr<ppc::core::TaskData> taskData;
  double a, b, epsilon;
  int cnt_of_splits;
  double h;
  double res;                        
  std::vector<double> input_;        
  double function_square(double x);  
};
}  // namespace golovkin_integration_rectangular_method