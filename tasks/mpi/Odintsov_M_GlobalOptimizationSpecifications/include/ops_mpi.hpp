#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace Odintsov_M_GlobalOptimizationSpecifications_mpi {

class GlobalOptimizationSpecificationsMPISequential : public ppc::core::Task {
 public:
  explicit GlobalOptimizationSpecificationsMPISequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  bool satisfies_constraints(double x, double y, int number_constraint);
  double calculate_function(double x, double y);
  double step;
  std::vector<double> area;   // �������� 4 ����� - ������� �������
  std::vector<double> funct;  // �������� 2 ����� (a,b) � ������� ������� ����� �������������� ������� (x-a)^2+(y-b)^2
  std::vector<double> constraint;  // ������ ����������� ����� �������������� 3 ������� (a,b,c) a * x + b * y - c
  int ver;
  int count_constraint;
  double ans;
};

class GlobalOptimizationSpecificationsMPIParallel : public ppc::core::Task {
 public:
  explicit GlobalOptimizationSpecificationsMPIParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  bool satisfies_constraints(double x, double y, int number_constraint);
  double calculate_function(double x, double y);
  double step;
  std::vector<double> area;   // �������� 4 ����� - ������� �������
  std::vector<double> funct;  // �������� 2 ����� (a,b) � ������� ������� ����� �������������� ������� (x-a)^2+(y-b)^2
  std::vector<double> constraint;  // ������ ����������� ����� �������������� 3 ������� (a,b,c) a * x + b * y - c
  std::vector<double> local_constraint;
  int ver;
  int count_constraint;
  double ans;

  boost::mpi::communicator com;
  int loc_constr_size;
  std::vector<int> is_corret;
};
}  // namespace Odintsov_M_GlobalOptimizationSpecifications_seq_mpi