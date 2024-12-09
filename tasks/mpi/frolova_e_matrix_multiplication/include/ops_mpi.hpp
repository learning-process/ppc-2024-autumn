#pragma once

#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>


#include <boost/serialization/vector.hpp>
#include <boost/serialization/access.hpp>

#include "core/task/include/task.hpp"

#include <boost/mpi.hpp>


namespace frolova_e_matrix_multiplication_mpi {

    void randomNumVec(int N, std::vector<int>& vec);
    std::vector<int> Multiplication(size_t M, size_t N, size_t K, const std::vector<int>& A, const std::vector<int>& B);

class matrixMultiplicationSequential : public ppc::core::Task {
 public:
  explicit matrixMultiplicationSequential(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrixA{};
  std::vector<int> matrixB{};
  std::vector<int> matrixC{};

  size_t lineA{};
  size_t columnA{};

  size_t lineB{};
  size_t columnB{};
};

struct lineStruc {
  std::vector<int> local_lines{};
  std::vector<int> index_lines{};
  size_t numberOfLines{};
  size_t enterLineslenght{};

  std::vector<int> res_lines{};
  size_t outgoingLineLength{};


  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& local_lines;
    ar& index_lines;
    ar& numberOfLines;
    ar& enterLineslenght;
    ar& res_lines;
    ar& outgoingLineLength;
  }

  lineStruc()
      : local_lines{},         
        index_lines{},         
        numberOfLines{0},      
        enterLineslenght{0},   
        res_lines{},           
        outgoingLineLength{0}  
  {}

  lineStruc& operator=(const lineStruc& other) {
    if (this != &other) {  
      local_lines = other.local_lines;
      index_lines = other.index_lines;
      numberOfLines = other.numberOfLines;
      enterLineslenght = other.enterLineslenght;
      res_lines = other.res_lines;
      outgoingLineLength = other.outgoingLineLength;
    }
    return *this;
  }

};

struct columnStruc {
  std::vector<int> local_columns{};
  std::vector<int> index_colums{};
  size_t numberOfColumns{};
  size_t enterColumnLenght{};


  template <class Archive>
  void serialize(Archive& ar, const unsigned int version) {
    ar& local_columns;
    ar& index_colums;
    ar& numberOfColumns;
    ar& enterColumnLenght;
  }

  columnStruc()
      : local_columns{},      
        index_colums{},       
        numberOfColumns{0},   
        enterColumnLenght{0}  
  {}

  columnStruc& operator=(const columnStruc& other) {
    if (this != &other) { 
      local_columns = other.local_columns;
      index_colums = other.index_colums;
      numberOfColumns = other.numberOfColumns;
      enterColumnLenght = other.enterColumnLenght;
    }
    return *this;
  }

};

void multiplyAndPlace(lineStruc& line, const columnStruc& column);

class matrixMultiplicationParallel : public ppc::core::Task {
 public:
  explicit matrixMultiplicationParallel(std::shared_ptr<ppc::core::TaskData> taskData_)
      : Task(std::move(taskData_)) {}
  bool pre_processing() override;
  bool validation() override;
  bool run() override;
  bool post_processing() override;

 private:
  std::vector<int> matrixA{};// only zero process
  std::vector<int> matrixB{};// only zero process
  std::vector<int> matrixC{};// only zero process

  lineStruc localLinesA{};
  columnStruc localColumnB{};

  size_t lineA{};
  size_t columnA{};

  size_t lineB{};
  size_t columnB{};

  boost::mpi::communicator world;
};

}  // namespace frolova_e_matrix_multiplication_mpi