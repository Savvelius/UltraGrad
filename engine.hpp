#pragma once

#include "size.hpp"
#include "smartptr.hpp"
#include <memory>
#include <cassert>
#include <initializer_list>
#include <ostream>
#include <array>
#include <functional>
#include "deprecated.hpp"   // shall be deleted soon

template<Algebraic T>
class Tensor {
public:
	Tensor() = default;
	Tensor(std::initializer_list<len_type>, T);
	Tensor(const Tensor<T>&, bool empty = false);
	Tensor(Tensor<T>&&) noexcept;
    explicit Tensor(const Size&);
    explicit Tensor(len_type);

    Comparison compare (const Tensor<T>&);
    T& item() const;
	[[nodiscard]] Size     shape() const;
    [[nodiscard]] dim_type dims()  const;
    [[nodiscard]] len_type numel(dim_type start_dim = 0) const;
    [[nodiscard]] len_type getOffset(std::initializer_list<idx_type>) const;

    Tensor<T> empty_like() = delete;
	Tensor<T>& operator = (const Tensor&);
	Tensor<T>& operator = (T);
	Tensor<T>& operator = (Tensor&&) noexcept;

    // NOTE: all these ops share pointer to data. Check for it.
    Tensor<T> operator [] (idx_type);       // Ready
	Tensor<T> operator [] (std::initializer_list<len_type>);    // no multiple arguments until c++23
    Tensor<T> operator [] (const Tensor<bool>&) = delete;       // THINK: maybe useless(move version is more popular)
    Tensor<T> operator [] (Tensor<bool>&&)      = delete;       // will be implemented after boolean ops
    [[deprecated]] Tensor<T> operator [] (Range<len_type>&&);// indexing array without creating new Range instance

    Tensor<bool> operator == (const Tensor<T>&) const = delete;
    Tensor<bool> operator == (T)                const = delete;
    Tensor<bool> operator > (const Tensor<T>&)  const = delete;
    Tensor<bool> operator > (T)                 const = delete;
    Tensor<bool> operator < (const Tensor<T>&)  const = delete;
    Tensor<bool> operator < (T)                 const = delete;

    Tensor<T>  bin_op(const Tensor<T>&, std::function<void(T&, T)>) const;
    Tensor<T>& bin_op_ip(const Tensor<T>&, std::function<void(T&, T)>);
    Tensor<T>  un_op(std::function<void(T&)>) const;
    Tensor<T>& un_op_ip(std::function<void(T&)>);

    Tensor<T> operator + (const Tensor<T>&) const;        // continue from here
    Tensor<T> operator + (T) const;
    Tensor<T> operator - (const Tensor<T>&) const;
    Tensor<T> operator - (T) const;
    Tensor<T> operator * (const Tensor<T>&) const;
    Tensor<T> operator * (T) const;
    Tensor<T> operator / (const Tensor<T>&) const;
    Tensor<T> operator / (T) const;
    // Those 3 aren't first priority. Thinking about details for now
    Tensor<T> operator ^ (double) const           = delete;  // analogue of __pow__
    Tensor<T> operator ^ (float) const            = delete;  // analogue of __pow__
    Tensor<T> operator ^ (int) const              = delete;  // analogue of __pow__

    Tensor<T> operator - () const;

    Tensor<T>& operator += (const Tensor<T>&);
    Tensor<T>& operator += (T);
    Tensor<T>& operator -= (const Tensor<T>&);
    Tensor<T>& operator -= (T);
    Tensor<T>& operator *= (const Tensor<T>&);
    Tensor<T>& operator *= (T);
    Tensor<T>& operator /= (const Tensor<T>&);
    Tensor<T>& operator /= (T);

    Tensor<T> relu();
    // NOTE: works properly only with float/double
    Tensor<T> e();
    Tensor<T> tanh();
    Tensor<T> sigmoid();
    Tensor<T> bmm(const Tensor<T>&) = delete;   // batch matrix multiplication

    // reductors
    Tensor<T> reduce_op(T&, std::function<void(T)>, dim_type dim = 0, bool for_all = true);  // generalization would be nice
    Tensor<T> sum(dim_type dim = 0, bool for_all = true);
    Tensor<T> max(dim_type dim = 0, bool for_all = true);
    Tensor<T> min(dim_type dim = 0, bool for_all = true);
    Tensor<T> argmax(dim_type dim = 0, bool for_all = true);
    Tensor<T> argmin(dim_type dim = 0, bool for_all = true);

    void backward() = delete;

	~Tensor();

    friend std::ostream& operator<<(std::ostream& out, const Tensor<T>& object) {
        //FIXME outputs tensor as 1d array
        // should probably be recursive
        out << '[';
        for (len_type i = 0; i < object.numel(0); i++){
            out << object.data[i] << ", ";
        }
        out << ']' << std::endl;
        return out;
    }
private:
//	  SharedPtr<T> data;
    T* data = nullptr;
    bool is_owner = true;
    Size size;
    // place for a lambda _backward function: std::function<void(...)> _backward = nullptr;
    // place for another T* holding gradient values: T* grad;
    // place for a boolean grad flag: bool requires_grad = false;
    // place for (some version of array) of parent Tensors.
    Tensor(T*&, Size&&);
};

// for now not implementing keepdim
template<Algebraic T>
Tensor<T> Tensor<T>::reduce_op(T& accumulate, std::function<void(T)> reduction, dim_type dim, bool for_all) {
    assert(dim < this->dims() && "Dimension out of bounds (reduce_op)");
    if (for_all) {
        std::for_each(data, data + numel(), reduction);
        Tensor<T> out(1);
        out.data[0] = accumulate;
        return out;
    }
    // implement dimension-wise
    Tensor<T> out(this->size.copy_except(dim));
    assert(0 && "not implemented");
    return out;
}

template<Algebraic T>
Tensor<T> Tensor<T>::sum(dim_type dim, bool for_all) {
    assert(0 && "Not implemented");
    T accumulate = 0;
    return this->reduce_op(accumulate, [&accumulate](T x)->void{ accumulate += x; }, dim, for_all);
}

template<Algebraic T>
Tensor<T> Tensor<T>::sigmoid() {
    return this->un_op([](T& x)->void{ x = 1 / (exp(-x) + 1); });
}

template<Algebraic T>
Tensor<T> Tensor<T>::tanh() {
    return this->un_op([](T& x)->void{ T temp = exp(2*x); x = (temp - 1) / (temp  +1); });
}

template<Algebraic T>
Tensor<T> Tensor<T>::e() {
    return this->un_op([](T& x)->void{ x = exp(x); });
}

template<Algebraic T>
Tensor<T> Tensor<T>::relu() {
    return this->un_op([](T& x)->void{x = (x>0)?x:0;});
}

template<Algebraic T>
Tensor<T> &Tensor<T>::operator/=(T other) {
    assert(other != 0);
    return this->un_op_ip([other](T& x)->void{ x/=other; });
}

template<Algebraic T>
Tensor<T>& Tensor<T>::operator/=(const Tensor<T> & other) {
    return this->bin_op_ip(other, [](T& x, T y)->void{ assert(y != 0);x /= y; });
}

template<Algebraic T>
Tensor<T>& Tensor<T>::operator*=(T other) {
    return this->un_op_ip([other](T& x)->void{ x*=other; });
}

template<Algebraic T>
Tensor<T> &Tensor<T>::operator*=(const Tensor<T> & other) {
    return this->bin_op_ip(other, [](T& x, T y)->void{x *= y;});
}

template<Algebraic T>
Tensor<T>& Tensor<T>::operator-=(T other) {
    return this->un_op_ip([other](T& x)->void{x -= other;});
}

template<Algebraic T>
Tensor<T>& Tensor<T>::operator-=(const Tensor<T> & other) {
    return this->bin_op_ip(other, [](T& x, T y)->void{ x -= y; });
}

template<Algebraic T>
Tensor<T>& Tensor<T>::operator+=(T other) {
    return this->un_op_ip([other](T& x)->void{ x += other; });
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator/(T other) const {
    assert(other != 0);
    return Tensor<T>(*this).un_op_ip([other](T& x)->void{ x /= other; });
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator / (const Tensor<T> & other) const {
    return this->bin_op(other, [](T& x, T y){ assert(y != 0); x /= y; });
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator*(T other) const {
    return this->un_op([other](T& x)->void{x *= other;});
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> & other) const {
    return this->bin_op(other, [](T& x, T y)->void{ x *= y; });
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator-(T other) const {
    return *this + (-other);
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> & other) const {
    return (*this + (-other));
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator+(T other) const {
    Tensor<T> out(*this);
    return out.un_op_ip([other](T& x)->void{ x += other; });
}

template<Algebraic T>
inline Tensor<T>& Tensor<T>::un_op_ip(std::function<void(T &)> operation) {
    std::for_each(data, data + numel(), operation);
    return *this;
}

template<Algebraic T>
Tensor<T>& Tensor<T>::operator=(T other) {
    std::fill(data, data + numel(), other);
    return *this;
}

template<Algebraic T>
Tensor<T>::~Tensor() {
    if (is_owner && data)
        delete[] data;
}

template<Algebraic T>
inline T& Tensor<T>::item() const {
    return data[0];
}

// FIXME: if compare operator isn't use anywhere else, inline it into this function and remove Compare enum class
template<Algebraic T>
inline Tensor<T>& Tensor<T>::bin_op_ip(const Tensor<T>& other, std::function<void(T&, T)> operation) {
    Comparison state = this->compare(other);
    T* out; len_type out_s;
    T* min_t; len_type min_s;
    switch (state) {
        case Comparison::eq: {
            util::apply_ip<T>(data, numel(), other.data, operation);
            return *this;
        }
        case Comparison::gt: {
            out   = this->data; out_s = this->numel();
            min_t = other.data; min_s = other.numel();
            break;
        }
        case Comparison::lt: {
            out   = other.data; out_s = other.numel();
            min_t = this->data; min_s = this->numel();
            break;
        }
        case Comparison::ne: {
            assert(0 && "Shapes of tensors don't match");
            break;
        }
        default: {
            assert(0 && "Can't happen");
        }
    }
    for (int block = 0; block < out_s; block += min_s)
        for (int i = 0; i < min_s; i++)
            operation(out[block + i], min_t[i]);
    return *this;
}

template<Algebraic T>
Comparison Tensor<T>::compare(const Tensor<T> & other) {
    return this->size.compare(other.size);
}

template<Algebraic T>
Tensor<T>::Tensor(const Size & new_size) {
    this->size = new_size;
    this->data = new T[numel()];
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator-() const {
    return this->un_op([](T& x) -> void{ x *= -1; });
}

template<Algebraic T>
Tensor<T>& Tensor<T>::operator+=(const Tensor<T> & other) {
    return this->bin_op_ip(other, [](T& mut, T arg) -> void{ mut += arg; });
}

template<Algebraic T>
inline Tensor<T> Tensor<T>::un_op(std::function<void(T&)> operation) const {
    Tensor<T> out(*this);
    return out.un_op_ip(operation);
}

// FIXME: implement a lot of helper - function and optimize the process, which is bloated for now
template<Algebraic T>
Tensor<T> Tensor<T>::bin_op(const Tensor<T> & other, std::function<void(T&, T)> operation) const {
    Tensor<T> out(*this);
    return out.bin_op_ip(other, operation);
}

template<Algebraic T>
Tensor<T> Tensor<T>::operator+(const Tensor<T> & other) const {
    return this->bin_op(other, [](T& x, T y) -> void { x += y; });
}

// FIXME: can't initialize data without initializing size
template<Algebraic T>
Tensor<T>::Tensor(len_type length) {
    this->data = new T[length];
    this->size = Size({length});
}

template<Algebraic T>
[[deprecated("will be removed soon")]] Tensor<T> Tensor<T>::operator[](Range<len_type> && index) {
    return *this;
    auto info = index.get_info();
    assert(info.start >= 0);
    assert(info.stop < this->size[0]);
    Tensor<T> t(index.size() * this->numel(1));

    return std::move(t);
}

// FIXME: checkme
template<Algebraic T>
Tensor<T> Tensor<T>::operator[](idx_type index) {
    assert(index < this->size[0]);
    T* p_out;
    if (this->dims() == 1)
        p_out = data + index;
    else
        p_out = data + index * numel(1);
    Tensor<T> out(p_out, Size(size, 1));
    return out;
}

template<Algebraic T>
inline dim_type Tensor<T>::dims() const {
    return this->size.dims();
}

// NOTE: no ownership
template<Algebraic T>
Tensor<T>::Tensor(T*& new_data, Size&& new_shape)
    : data{new_data}, size{new_shape} {
    this->is_owner = false;
    new_data = nullptr;
}

template<Algebraic T>
inline  len_type Tensor<T>::getOffset(std::initializer_list<idx_type> args) const {
    //NOTE: don't really need dim checking(for now it's called only from [{...}] operator)
    if (args.size() == 1){
        return numel(1) * (*args.begin());
    }
    len_type result = 0;
    idx_type i = 1;     //args.size();
    for (auto item = args.begin(); item < args.end() - 1; ++item, ++i){
        result += (*item) * numel(i);
    }
    result += *(args.end() - 1);
    return result;
}

template<Algebraic T>
inline Tensor<T>::Tensor(std::initializer_list<len_type> shape, T fill_value) {
	this->size    = shape;
	len_type prod = this->numel(0);
	this->data = new T[prod];
//    std::fill(data, data + prod, fill_value);
	for (len_type i = 0; i < prod; i++, fill_value++) data[i] = fill_value;
}

template<Algebraic T>
inline Tensor<T>::Tensor(const Tensor& other, bool empty) {
	this->data = new T[other.numel()];
    this->size = other.size;
    if (!empty)
        std::copy(other.data, other.data + numel(), data);
}

template<Algebraic T>
inline Tensor<T>::Tensor(Tensor&& other) noexcept
    : data{other.data}, size{std::move(other.size)} {
    other.data = nullptr;
}

// for now useless, to be implemented
template<Algebraic T >
inline Tensor<T>& Tensor<T>::operator=(const Tensor& other) {
    if (this == &other)
        return *this;
    if (this->data) {
        if (this->size == other.size)
            goto copy_data;
        delete[] this->data;
    }
    this->size = other.size;
    this->data = new T[numel()];
    copy_data:
        std::copy(other.data, other.data + numel(), this->data);
	return *this;
}

template<Algebraic T>
inline Tensor<T>& Tensor<T>::operator=(Tensor&& other) noexcept {
    if (this->data)
        delete[] this->data;
    this->data = other.data;
    this->size = std::move(other.size);
    other.data = nullptr;
	return *this;
}

template<Algebraic T>
inline Tensor<T> Tensor<T>::operator[](std::initializer_list<len_type> indexes)  {
	assert(this->dims() != 0 && "can't subscript scalar tensor");
	assert(indexes.size() <= this->dims() && "too many indexes to subscript");
	//FIXME, horrible match checking
    len_type arg_size = indexes.size();
    assert(size > indexes && "Not compatible sizes");
    T* p_out = this->data + this->getOffset(indexes);
    Size s_out(this->size, arg_size);
	return Tensor<T>(p_out, std::move(s_out));
}

template<Algebraic T>
inline Size Tensor<T>::shape() const {
	return this->size;
}

template<Algebraic T>
inline len_type Tensor<T>::numel(dim_type start_dim) const {
	assert(start_dim < this->dims() && "dim number is too high (numel)");
	len_type prod = 1;
	for (len_type i = start_dim; i < this->dims(); ++i) prod *= this->size[i];
	return prod;
}

