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

void callable(int& accum, std::function<void(int)> op) {
    int arr[2][5] = {1,2,3,4,5,6,7,8,9,10};
    int maxes[2] = {0};
    auto temp = accum;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 5; ++j) {
            op(arr[i][j]);
        }
        maxes[i] = accum;
        accum = temp;
    }
    cout << "maxes[0] = " << maxes[0] << " | maxes[1] = " << maxes[1] << endl;
}

void caller() {
    int acc = 0;
    callable(acc, [&acc](int x)->void{acc += x;});
}

int main() {
    Tensor<float> t1({2, 2, 3}, 0.0);
    cout << t1 << t1.sum(2, false);
//    caller();
}
















