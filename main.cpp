#include <iostream>
#include <typeinfo>
#include "engine.hpp"
#include "utils.hpp"
#include <cstdarg>
#include <initializer_list>
#include <vector>
#include <algorithm>
#include <optional>
#include <concepts>
#include <array>
#include <span>

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
    int id = 0;
public:
    Test(int x = 0) {id = x; cout << "Def constr " << id << endl;}
    ~Test() {cout << "Destr " << id << endl;}
    Test(Test&& t) noexcept {
        cout << "move constr " << id << endl;
        id = t.id;
        t.id = 0;
    }
    Test(const Test& t){
        cout << "copy constr " << id << endl;
        this->id = t.id;
    }
    Test& operator=(const Test& other){
        cout << "Copy assign " << id << endl;
        return *this;
    }
    Test& operator=(Test&& other){
        cout << "Move assign " << id << endl;
        return *this;
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
//    Tensor<int> t1({10}, 0);
//    auto t2 = t1;
//    t2[1].item() = 100;
//    cout << t1;

}


