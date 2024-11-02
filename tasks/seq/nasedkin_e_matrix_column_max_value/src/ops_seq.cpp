#include "ops_seq.hpp"

namespace nasedkin_e_matrix_column_max_value {

    std::vector<int> getMaxValuesPerColumn(const std::vector<std::vector<int>>& matrix) {
        int numRows = matrix.size();
        int numCols = matrix[0].size();
        std::vector<int> maxValues(numCols, INT_MIN);

        for (int j = 0; j < numCols; ++j) {
            for (int i = 0; i < numRows; ++i) {
                maxValues[j] = std::max(maxValues[j], matrix[i][j]);
            }
        }
        return maxValues;
    }

}
