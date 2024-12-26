#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/serialization/serialization.hpp>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"

namespace voroshilov_v_bivariate_optimization_by_area_mpi {

struct Point {
  double x;
  double y;

  Point(double x_ = 0, double y_ = 0) {
    x = x_;
    y = y_;
  }

  template <class Archive>
  void serialize(Archive& ar, unsigned int version) {
    ar & x;
    ar & y;
  }
};

struct PointWithValue {
  double x;
  double y;
  double value;

  PointWithValue(double x_ = 0, double y_ = 0, double value_ = 0) {
    x = x_;
    y = y_;
    value = value_;
  }

  template <class Archive>
  void serialize(Archive& ar, unsigned int version) {
    ar & x;
    ar & y;
    ar & value;
  }
};

struct Monomial {
  double coef;
  int deg_x;
  int deg_y;

  Monomial(double coef_, int deg_x_, int deg_y_) {
    coef = coef_;
    deg_x = deg_x_;
    deg_y = deg_y_;
  }

  Monomial(std::vector<char> monom) {
    coef = 0.0;
    deg_x = 0;
    deg_y = 0;
    size_t i = 0;
    std::string str_coef;
    while ((i < monom.size()) && (monom[i] != 'x')) {
      str_coef += monom[i];
      i++;
    }
    if (!str_coef.empty()) {
      if (str_coef == "-") {
        coef = -1.0;
      } else if (str_coef == "+") {
        coef = 1.0;
      } else {
        coef = std::stod(str_coef);
      }
    }
    if (str_coef.empty()) {
      coef = 1.0;
    }

    if (i < monom.size()) {
      i += 2;
      std::string str_degx;
      while (monom[i] != 'y') {
        str_degx += monom[i];
        i++;
      }
      deg_x = std::stoi(str_degx);

      i += 2;
      std::string str_degy;
      while (i < monom.size()) {
        str_degy += monom[i];
        i++;
      }
      deg_y = std::stoi(str_degy);
    }
  }

  double calculate(Point point) const {
    double res = coef * pow(point.x, deg_x) * pow(point.y, deg_y);
    return res;
  }
};

struct Polynomial {
  size_t length;
  std::vector<Monomial> monomials;

  Polynomial() { length = 0; }

  Polynomial(std::vector<char> polynom) {
    length = 0;
    size_t i = 0;
    while (i < polynom.size()) {
      std::vector<char> monom;
      monom.push_back(polynom[i]);
      i++;
      while ((i < polynom.size()) && (polynom[i] != ' ')) {
        monom.push_back(polynom[i]);
        i++;
      }
      i++;  // skip ' '
      Monomial mnm(monom);
      monomials.push_back(mnm);
      length++;
    }
  }

  double calculate(Point point) {
    double res = 0.0;
    for (size_t i = 0; i < length; i++) {
      res += monomials[i].calculate(point);
    }
    return res;
  }
};

struct Search_area {
  double min_value;
  double max_value;
  int steps_count;

  Search_area(double min_value_ = 0, double max_value_ = 0, int steps_count_ = 0) {
    min_value = min_value_;
    max_value = max_value_;
    steps_count = steps_count_;
  }
};

class OptimizationMPITaskSequential : public ppc::core::Task {
 public:
  explicit OptimizationMPITaskSequential(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  Polynomial q;               // criterium function
  std::vector<Polynomial> g;  // constraints functions

  Search_area x_area;
  Search_area y_area;

  Point optimum_point;
  double optimum_value;
};

class OptimizationMPITaskParallel : public ppc::core::Task {
 public:
  explicit OptimizationMPITaskParallel(std::shared_ptr<ppc::core::TaskData> taskData_) : Task(std::move(taskData_)) {}
  bool validation() override;
  bool pre_processing() override;
  bool run() override;
  bool post_processing() override;

 private:
  Polynomial q;               // criterium function
  std::vector<Polynomial> g;  // constraints functions

  Search_area x_area;
  Search_area y_area;

  PointWithValue local_optimum_point;

  boost::mpi::communicator world;
};

}  // namespace voroshilov_v_bivariate_optimization_by_area_mpi