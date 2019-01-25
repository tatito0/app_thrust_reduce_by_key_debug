#pragma once

#include "picojson.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <fstream>

typedef thrust::pair<int, int> IdxPair;
#define INDEX_EL_IN_MAT(num_cols, row_index, col_index) ((row_index)*(num_cols)+(col_index))

struct PBA_Vec6
{
  double x;
  double y;
  double z;
  double a;
  double b;
  double c;
};

struct PBA_Mat6
{
  PBA_Vec6 r0;
  PBA_Vec6 r1;
  PBA_Vec6 r2;
  PBA_Vec6 r3;
  PBA_Vec6 r4;
  PBA_Vec6 r5;
};

__device__ __host__
inline void pba_mat_add(double* retVal, double* matA, double* matB, int m, int n)
{
  int i, j;
  for (i = 0; i < m; ++i)
  {
    for (j = 0; j < n; ++j)
    {
      retVal[INDEX_EL_IN_MAT(n, i, j)] = matA[INDEX_EL_IN_MAT(n, i, j)] + matB[INDEX_EL_IN_MAT(n, i, j)];
    }
  }
}

__device__ __host__
inline PBA_Mat6 operator+(const PBA_Mat6& lhs, const PBA_Mat6& rhs)
{
    PBA_Mat6 retVal;
    pba_mat_add((double*)&retVal, (double*)&lhs, (double*)&rhs, 6, 6);
    return retVal;
}

__device__ __host__
inline bool pba_mat_equal(double* matA, double* matB, int m, int n)
{
   int i, j;
   for (i = 0; i < m; ++i)
   {
       for (j = 0; j < n; ++j)
       {
           double valA = matA[INDEX_EL_IN_MAT(n, i, j)];
           double valB = matB[INDEX_EL_IN_MAT(n, i, j)];
           double error = fabs(valA - valB);
           bool eq = error < 1e-5;
           if(!eq)
               return false;
       }
   }

   return true;
}

__device__ __host__
inline bool operator==(const PBA_Mat6& lhs, const PBA_Mat6& rhs)
{
    return pba_mat_equal((double*)&lhs, (double*)&rhs, 6, 6);
}

__device__ __host__
inline bool operator!=(const PBA_Mat6& lhs, const PBA_Mat6& rhs)
{
    return !pba_mat_equal((double*)&lhs, (double*)&rhs, 6, 6);
}

__host__
inline void pba_mat_print(std::ostream& target, double* mat, int rows, int cols)
{
    for(int i = 0; i < rows; i++)
    {
        for(int j = 0; j < cols; j++)
        {
            target << mat[INDEX_EL_IN_MAT(cols, i, j)];
            if(j != (cols - 1))
            {
                target << ", ";
            }
        }
        target << std::endl;
    }
}

__host__
std::ostream& operator<<(std::ostream& target, const PBA_Mat6& source)
{
    pba_mat_print(target, (double*)&source, 6, 6);
    return target;
}

inline picojson::value Pair2Json(const IdxPair& pair)
{
    picojson::value jsonVal;

    jsonVal["first"] = pair.first;
    jsonVal["second"] = pair.second;

    return jsonVal;
}

inline IdxPair Json2Pair(const picojson::value& jsonVal)
{
    IdxPair pair;
    pair.first = jsonVal["first"].get<int64_t>();
    pair.second= jsonVal["second"].get<int64_t>();
    return pair;
}

inline picojson::value Mat2Json(const PBA_Mat6& mat)
{
    const double* matPtr = (double*)&mat;
    picojson::value jsonMat;
    for (int i = 0; i < 6; ++i)
    {
        picojson::value jsonRow;
        for (int j = 0; j < 6; ++j)
        {
            jsonRow.push_back(matPtr[INDEX_EL_IN_MAT(6, i, j)]);
        }
        jsonMat.push_back(jsonRow);
    }
    return jsonMat;
}

inline PBA_Mat6 Json2Mat(const picojson::value& jsonMat)
{
    PBA_Mat6 mat;
    double* matPtr = (double*)&mat;
    for (int i = 0; i < 6; ++i)
    {
        picojson::value jsonRow = jsonMat[i];
        for (int j = 0; j < 6; ++j)
        {
            matPtr[INDEX_EL_IN_MAT(6, i, j)] = jsonRow[j].get<double>();
        }
    }
    return mat;
}


inline bool SaveDebugData(const std::string& filePath, const thrust::device_vector<IdxPair>& d_pairArr, const thrust::device_vector<PBA_Mat6>& d_matArr)
{
    picojson::value jsonRoot;
    jsonRoot["itemCount"] = d_pairArr.size();

    //Serialize cameras
    picojson::value jsonKeyArr;
    picojson::value jsonValueArr;
    for(size_t i = 0; i <  d_pairArr.size(); ++i)
    {
        const IdxPair& pair = d_pairArr[i];
        const PBA_Mat6& mat = d_matArr[i];
        picojson::value jsonPair = Pair2Json(pair);
        picojson::value jsonMat = Mat2Json(mat);
        jsonKeyArr.push_back(jsonPair);
        jsonValueArr.push_back(jsonMat);
    }
    jsonRoot["KeyArr"] = jsonKeyArr;
    jsonRoot["MatArr"] = jsonValueArr;

    std::string jsonStr = jsonRoot.serialize(true);

    bool saveOk = false;
    std::ofstream outFile;
    outFile.open(filePath);
    if(outFile.is_open())
    {
        outFile << jsonStr;
        saveOk = true;
        std::cout << "Data saved to: " << filePath << std::endl;
    }
    else
    {
        std::cout << "Error: Failed to save data to: " << filePath << std::endl;
    }
    outFile.close();

    return saveOk;
}

inline bool LoadDebugData(const std::string& filePath, thrust::host_vector<IdxPair>& h_pairArr, thrust::host_vector<PBA_Mat6>& h_matArr)
{
    std::ifstream inFile(filePath);
    if (!inFile.is_open())
    {
        std::cout << "Error: Failed to open data file: " << filePath << std::endl;
        return false;
    }

    picojson::value jsonRoot;
    std::string err = picojson::parse(jsonRoot, inFile);
    if (!err.empty())
    {
        std::cout << "Error: Failed to parse data file: " << filePath << std::endl;
        return false;
    }

    int itemCount = jsonRoot.get_value<int64_t>("itemCount", 0);
    h_pairArr.resize(itemCount);
    h_matArr.resize(itemCount);

    picojson::value jsonKeyArr = jsonRoot["KeyArr"];
    for(int i = 0; i <  itemCount; ++i)
    {
        IdxPair pair = Json2Pair(jsonKeyArr[i]);
        h_pairArr[i] = pair;
    }

    picojson::value jsonValuArr = jsonRoot["MatArr"];
    for(int i = 0; i <  itemCount; ++i)
    {
        PBA_Mat6 mat = Json2Mat(jsonValuArr[i]);
        h_matArr[i] = mat;
    }

    return true;
}
