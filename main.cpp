#include <iostream>
#include <typeinfo>
#include "engine.hpp"
#include "utils.hpp"
#include <cstdarg>
#include <initializer_list>
#include <vector>
#include <algorithm>

using std::cout, std::endl, std::vector;

/*
TODO: replace all asserts with exceptions or something else
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

class Test {
private:
    int x = 0;
public:
    Test() {cout << "Def constr" << endl;}
    ~Test() {cout << "Destr" << endl;}
    Test(Test&& t) noexcept {
        cout << "move constr" << endl;
        x = t.x;
        t.x = 0;
    }
    Test(const Test& t){
        cout << "copy constr" << endl;
        this->x = t.x;
    }
};


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

constexpr int sum(int first, int second){
    return first + second;
}

template<typename T, uint32_t size>
constexpr T arrSum(std::array<T, size> arr){
    T out = 0;
    for (int i = 0; i < arr.size(); i++) out += arr[i];
    return out;
}

SharedPtr<int> f(SharedPtr<int> p){
    return p;
}

int main() {
    RangeIterator it;
    ++ it;
    it++;
}
