#pragma once

#include <gtest/gtest.h>

#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace frolova_e_Simpson_method_mpi {

    

    // one function
double squaresOfX(const std::vector<double>& point, int dimension);
double cubeOfX(const std::vector<double>& point, int dimension);

// two functions
double sumOfSquaresOfXandY(const std::vector<double>& point, int dimension);
double ProductOfXAndY(const std::vector<double>& point, int dimension);

// three functions
double sumOfSquaresOfXandYandZ(const std::vector<double>& point, int dimension);
double ProductOfSquaresOfXandYandZ(const std::vector<double>& point, int dimension);

//_______________________________________________________________________________________________________________

static std::map<int, double (*)(const std::vector<double>&, int)> functionRegistry = {{1, squaresOfX},
                                                                                      {2, cubeOfX},
                                                                                      {3, sumOfSquaresOfXandY},
                                                                                      {4, ProductOfXAndY},
                                                                                      {5, sumOfSquaresOfXandYandZ},
                                                                                      {6, ProductOfSquaresOfXandYandZ}};

//________________________________________________________________________________________________________________


double roundToTwoDecimalPlaces(double value);

double Simpson_Method(double (*func)(const std::vector<double>&, int), int divisions, int dimension,
                      std::vector<double>& limits);

class SimpsonmethodSequential : public ppc::core::Task {
 public:
  explicit SimpsonmethodSequential(std::shared_ptr<ppc::core::TaskData> taskData_,
                                   double (*func_)(const std::vector<double>&, int))
      : Task(std::move(taskData_)), func(func_) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  double (*func)(const std::vector<double>&, int);
  std::vector<double> limits;
  size_t divisions;
  size_t dimension;
  double resIntegral;
};

class SimpsonmethodParallel : public ppc::core::Task {
 public:
  explicit SimpsonmethodParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:

  double (*func)(const std::vector<double>&, int);

  //for first process
  std::vector<double> limits;
  size_t divisions;
  double resIntegral;

  //for all processes
  int functionid;

  std::vector<double> localLimits;
  size_t localdivisions;
  size_t dimension;

  double localres;


  boost::mpi::communicator world;
};

}  // namespace frolova_e_Simpson_method_mpi