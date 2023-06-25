#pragma once
#include "utils.hpp"
#include <memory>
#include <cassert>
#include <initializer_list>
#include <ostream>
#include <array>

template<typename T>
class Tensor {
public:
	Tensor() = default;
	Tensor(std::initializer_list<len_type>, T);
	Tensor(const Tensor<T>&);
	Tensor(Tensor<T>&&) noexcept;

	Size     shape() const;
	dim_type dims()  const;
    len_type numel(dim_type) const;
    len_type getOffset(std::initializer_list<idx_type>) const;

	Tensor<T>& operator = (const Tensor&);
	Tensor<T>& operator = (Tensor&&) noexcept;

    Tensor<T> operator [] (idx_type) const = delete;
	Tensor<T> operator [] (std::initializer_list<idx_type>) const;    // no multiple arguments until c++23
    Tensor<T> operator [] (const Tensor<bool>&) const = delete;                // returns a 1d Tensor
    Tensor<T> operator [] (Range<len_type>&&) = delete;                                  // indexing array without creating new Range instance

    Tensor<bool> operator == (const Tensor<T>&) = delete;
    Tensor<bool> operator > (const Tensor<T>&) = delete;
    Tensor<bool> operator < (const Tensor<T>&) = delete;

	~Tensor() = default;

    template<typename U>
    friend std::ostream& operator << (std::ostream&, const Tensor<U>&);
private:
	SharedPtr<T> data;  // FIXME check for compatibility
    Size size;
	Tensor(T* , Size);
};

template<typename T>
dim_type Tensor<T>::dims() const {
    return this->size.length;
}

template<typename T>
Tensor<T>::Tensor(T *new_data, Size new_shape): data(new_data), size(new_shape) {
    this->_dims = this->_shape.size();
}

template<typename T>
inline  len_type Tensor<T>::getOffset(std::initializer_list<idx_type> args) const {
    //FIXME will not check for shape matches
    if (args.size() == 1){
        return numel(1);
    }
    len_type result = 0;
    idx_type i = 1;//args.size();
    for (auto item = args.begin(); item < args.end() - 1; item++){
        result += (*item) * numel(i);
        i++;
    }
    result += *(args.end() - 1);
    return result;
}

template<typename T>
inline Tensor<T>::Tensor(std::initializer_list<len_type> shape, T fill_value) {
	this->size = shape;
	len_type prod = this->numel(0);
	this->data = new T [prod];
	for (len_type i = 0; i < prod; i++, fill_value++) data[i] = fill_value;
}

template<typename T>
inline Tensor<T>::Tensor(const Tensor& other) {
	this->data = new T[other.numel(0)];
	memcpy(this->data, other.data, other.numel(0));
	this->size(other.size);
}

template<typename T>
inline Tensor<T>::Tensor(Tensor&& other) noexcept {
	this->data = other.data;
	this->size = other.size;

	other.data = nullptr;
    delete other.size; //FIXME maybe
}

template<typename T >
inline Tensor<T>& Tensor<T>::operator=(const Tensor& other)
{
	return *this;
}

template<typename T>
inline Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
	// for now useless, to be implemented
	return *this;
}

// TOP PRIORITY for now
template<typename T>
inline Tensor<T> Tensor<T>::operator[](std::initializer_list<idx_type> indexes) const {
	assert(this->dims() != 0 && "can't subscript scalar tensor");
	assert(indexes.size() <= this->_dims && "too many indexes to subscript");
	//FIXME, horrible match checking
    idx_type arg_size = 0;
    for (len_type item : indexes) {
		assert(item < this->size[arg_size] && "index out of bounds");
	    arg_size++;
    }
    // returning
	return Tensor<T> ((this->data) + this->getOffset(indexes), Size(this->size, arg_size));
}

template<typename T>
inline Size Tensor<T>::shape() const {
	return this->size;
}

template<typename T>
std::ostream& operator<<(std::ostream& out, const Tensor<T>& object) {
    // outputs tensor as 1d array FIXME
    out << '[';
    for (len_type i = 0; i < object.numel(0); i++){
        out << object.data[i] << ", ";
    }
    out << ']' << std::endl;
    return out;
}

template<typename T>
inline len_type Tensor<T>::numel(dim_type start_dim) const {
	assert(start_dim < this->dims() && "dim number is too high");
	len_type prod = 1;
	for (len_type i = start_dim; i < this->dims(); ++i) prod *= this->size[i];
	return prod;
}

