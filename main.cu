#include "utils.h"


int main()
{
    thrust::host_vector<IdxPair> h_pairArr;
    thrust::host_vector<PBA_Mat6> h_matArr;
    LoadDebugData("../test.json", h_pairArr, h_matArr);

    //Reduce in CPU
    typedef thrust::host_vector<IdxPair>::iterator H_FirstIt;
    typedef thrust::host_vector<PBA_Mat6>::iterator H_SecondIt;
    thrust::pair<H_FirstIt, H_SecondIt> h_newEnd;

    thrust::host_vector<IdxPair> h_reducedPairArr(h_pairArr.size());
    thrust::host_vector<PBA_Mat6> h_reducedMatArr(h_matArr.size());
    h_newEnd = thrust::reduce_by_key(h_pairArr.begin(), h_pairArr.end(), h_matArr.begin(), h_reducedPairArr.begin(), h_reducedMatArr.begin());

    int h_reducedCount = h_newEnd.first - h_reducedPairArr.begin();
    h_reducedPairArr.resize(h_reducedCount);
    h_reducedMatArr.resize(h_reducedCount);

    //Reduce in GPU
    thrust::device_vector<IdxPair> d_pairArr = h_pairArr;
    thrust::device_vector<PBA_Mat6> d_matArr = h_matArr;

    typedef thrust::device_vector<IdxPair>::iterator D_FirstIt;
    typedef thrust::device_vector<PBA_Mat6>::iterator D_SecondIt;
    thrust::pair<D_FirstIt, D_SecondIt> d_newEnd;

    thrust::device_vector<IdxPair> d_reducedPairArr(d_pairArr.size());
    thrust::device_vector<PBA_Mat6> d_reducedMatArr(d_matArr.size());
    d_newEnd = thrust::reduce_by_key(d_pairArr.begin(), d_pairArr.end(), d_matArr.begin(), d_reducedPairArr.begin(), d_reducedMatArr.begin());

    int d_reducedCount = d_newEnd.first - d_reducedPairArr.begin();
    d_reducedPairArr.resize(d_reducedCount);
    d_reducedMatArr.resize(d_reducedCount);

    //Compare results
    bool passed = true;
    if(h_reducedCount != d_reducedCount)
    {
        std::cout << "Error: reduced count don't match" << std::endl;
        passed = false;
    }
    else
    {
        for(int i = 0; i < h_reducedCount; ++i)
        {
            IdxPair h_pair = h_reducedPairArr[i];
            PBA_Mat6 h_mat = h_reducedMatArr[i];

            IdxPair d_pair = d_reducedPairArr[i];
            PBA_Mat6 d_mat = d_reducedMatArr[i];

            if(h_pair != d_pair)
            {
                std::cout << "Error: pair don't match at " << i << std::endl;
                std::cout << "host pair: " << i << " (" << h_pair.first << ", " << h_pair.second << ")" << std::endl;
                std::cout << "dev  pair: " << i << " (" << d_pair.first << ", " << d_pair.second << ")" << std::endl;
                passed = false;
            }

            if(h_mat != d_mat)
            {
                std::cout << "Error: mat don't match at " << i << std::endl;
                std::cout << "host mat: " << i << ":\n" << h_mat << std::endl;
                std::cout << "dev  mat: " << i << ":\n" << d_mat << std::endl;
                passed = false;
            }
        }
    }

    if(passed)
    {
        std::cout << "Results are correct" << std::endl;
    }
    else
    {
        std::cout << "Error: Results don't match" << std::endl;
    }

    return 0;
}
