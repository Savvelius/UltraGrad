#include <iostream>
#include <typeinfo>
#include "engine.hpp"
#include "utils.hpp"
#include <cstdarg>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <optional>
#include "timer.hpp"
#include <array>
#include <span>

using std::cout, std::endl, std::vector;

/*
TODO: replace all asserts with exceptions or something else
TODO: in all divisions by zero/logs of zero replace asserts with infinity (use std::numeric_limits<T>::infinity)
*/
template<typename T>
void printVec(vector<T> vec) {
    cout << '[';
	for (auto el : vec) {
		cout << el << ", ";
	}
	cout << ']' << std::endl;
}
template<typename T, uint32_t size>
void printArr(T arr[size]){
    cout << '[';
    for (uint32_t i = 0; i < size; i++) {
        cout << arr[i] << ", ";
    }
    cout << ']' << std::endl;
}

class Matrix {
private:
    std::array<int, 2> shape;
    int* data = nullptr;
public:
    Matrix(std::array<int, 2> shape_, int start_val){
        shape = shape_;
        int prod = shape[0] * shape[1];
        data = new int[prod];
        for (int i = 0; i < prod; i++){
            data[i] = start_val;
            start_val++;
        }
    }
    friend std::ostream& operator << (std::ostream& out, const Matrix& m) {
        out << '[';
        for (int i = 0; i < m.shape[0]; i++){
            out << '[';
            for (int j = 0; j < m.shape[1]; j++){
                out << m.data[i * m.shape[1] + j] << ", ";
            }
            out << "],\n";
        }
        out << ']' << endl;
        return out;
    }
};
class Ptr {
    int* memory = nullptr;
    bool is_owner = true;
public:
    Ptr() = default;
    Ptr(size_t size) {
        memory = new int[size];
    }
    Ptr(const Ptr& other);              // deep copy
    Ptr(Ptr&& other);                   // shallow copy
    Ptr& operator=(const Ptr& other);   // deep copy
    Ptr& operator=(Ptr&& other);        // shallow copy
    ~Ptr();
};

int main() {
    Tensor<float> t1({2, 200, 1000}, 0.0);
//    cout << t1[{1, 1}] << t1.reshape({2, 2, 2}).shape() << t1;
//    cout << t1.max(2);
    constexpr int NUM_ITERS = 1;
    Tensor<float> result;
    int time1 = 0, time2 = 0;
    {
        globals::CPU_MULTITHREAD = true;
        Timer tm("Parallel");
        for (int _ = 0; _ < NUM_ITERS; ++_)
            result = t1.max();
        time2 += tm.get_time();
    }
    {
        globals::CPU_MULTITHREAD = false;
        Timer tm("Sequential");
        for (int _ = 0; _ < NUM_ITERS; ++_)
            result = t1.max();
        time1 += tm.get_time();
    }
    cout << "Average sequnetial time = " << time1 << endl
         << "Average parallel time = "   << time2;
 }
















