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

    [[nodiscard]] Comparison compare (const Tensor<T>&) const;
    [[nodiscard]] T& item() const;
    [[nodiscard]] T& at(len_type) const;
	[[nodiscard]] Size     shape() const;
    [[nodiscard]] dim_type dims()  const;
    [[nodiscard]] len_type numel(dim_type start_dim = 0) const;
    [[nodiscard]] bool is_scalar() const;
    [[nodiscard]] len_type get_num_threads() const;     // Implement more sophisticated alg
    [[nodiscard]] bool all() const;

    // Those two are for range compatibility -> span usage
    ContiguousIterator<T> begin() const;
    ContiguousIterator<T> end() const;

    Tensor<T> empty_like() const = delete;
	Tensor<T>& operator = (const Tensor&);
	Tensor<T>& operator = (T);
	Tensor<T>& operator = (Tensor&&) noexcept;

    Tensor<T> operator [] (idx_type) const;       // Ready
	Tensor<T> operator [] (std::initializer_list<len_type>) const;    // no multiple arguments until c++23
    Tensor<T> operator [] (const Tensor<bool>&) const = delete;       // maybe useless(move version is more popular)
    Tensor<T> operator [] (Tensor<bool>&&)      const = delete;       // will be implemented after boolean ops
    [[deprecated]] Tensor<T> operator [] (Range<len_type>&&) const = delete;       // Deprecated custom class, replace with built-in range

    Tensor<bool> operator == (const Tensor<T>&) const;
    Tensor<bool> operator == (T)                const = delete;
    Tensor<bool> operator > (const Tensor<T>&)  const = delete;
    Tensor<bool> operator > (T)                 const = delete;
    Tensor<bool> operator < (const Tensor<T>&)  const = delete;
    Tensor<bool> operator < (T)                 const = delete;

    // NOTE: might be beneficial to change them to rvalue refs
    Tensor<T>  bin_op(const Tensor<T>&, const std::function<void(T&, T)>&) const;
    Tensor<T>& bin_op_ip(const Tensor<T>&, const std::function<void(T&, T)>&);
    Tensor<T>  un_op(const std::function<void(T&)>&) const;
    Tensor<T>& un_op_ip(const std::function<void(T&)>&);

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

    Tensor<T> relu() const;
    // NOTE: works properly only with float-like numbers
    Tensor<T> e() const;
    Tensor<T> tanh() const;
    Tensor<T> sigmoid() const;
    Tensor<T> matmul(const Tensor<T>&) const;   // batch matrix multiplication

    // reductors
    Tensor<T> reduce_op(T&, const std::function<void(T)>&, dim_type dim = 0, bool for_all = false, bool keepdim = false) const;  // generalization would be nice
    Tensor<T> sum(dim_type dim = 0, bool for_all = false, bool keepdim = false) const;
    Tensor<T> product(dim_type dim = 0, bool for_all = false, bool keepdim = false) const;
    // TODO: implement those two with special return types that contain indices(argmax/argmin) for backprop
    Tensor<T> max(dim_type dim = 0, bool for_all = false, bool keepdim = false) const;
    Tensor<T> min(dim_type dim = 0, bool for_all = false, bool keepdim = false) const;
    Tensor<len_type> argmax(dim_type dim = 0, bool for_all = true, bool keepdim = false) const = delete;
    Tensor<len_type> argmin(dim_type dim = 0, bool for_all = true, bool keepdim = false) const = delete;

    // Next functions just change the size object - trivial
    Tensor<T> unsqueeze(dim_type) const;
    Tensor<T> squeeze(dim_type) const;
    Tensor<T> reshape(std::initializer_list<len_type>) const;
    // Next are non-trivial functions
    Tensor<T> permute(std::initializer_list<dim_type>) const;
    Tensor<T> transpose() const;

    // Next are functions that mutate and concatenate tensors
    Tensor<T> concatenate(const Tensor<T>&, dim_type) const;

    void backward() = delete;

	~Tensor();

    friend Tensor<bool>;
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
    Tensor(T*, Size&&, bool is_owner_);   // FIXME: wtf is wrong with this constructor???
};

template<Algebraic T>
Tensor<T> Tensor<T>::reshape(std::initializer_list<len_type> new_shape) const {
    len_type start = 1;
    std::for_each(new_shape.begin(), new_shape.end(),
                  [&start](len_type x)->void{ start *= x; });
    assert(start == numel());
    return Tensor<T>(data, Size(new_shape), true);
}

template<Algebraic T>
Tensor<T> Tensor<T>::unsqueeze(dim_type dim) const {
    assert(dim < dims());
    return Tensor<T>(data, size.insert(dim, 1));
}

template<Algebraic T>
bool Tensor<T>::is_scalar() const {
    return size.is_scalar();
}

template<Algebraic T>
ContiguousIterator<T> Tensor<T>::end() const {
    return { this->data + numel() };
}
template<Algebraic T>
ContiguousIterator<T> Tensor<T>::begin() const {
    return { this->data };
}

template<Algebraic T>
Tensor<T> Tensor<T>::squeeze(dim_type dim) const {
    assert(size[dim] == 1);
    return Tensor<T>(data, size.remove(dim));
}

template<Algebraic T>
Tensor<T> Tensor<T>::min(dim_type dim, bool for_all, bool keepdim) const {
    T accumulate = std::numeric_limits<T>::max();
    return this->reduce_op(accumulate, [&accumulate](T x)->void{ accumulate = MIN(accumulate, x); },dim, for_all, keepdim);
}

template<Algebraic T>
Tensor<T> Tensor<T>::max(dim_type dim, bool for_all, bool keepdim) const {
    T accumulate = std::numeric_limits<T>::min();
    return this->reduce_op(accumulate, [&accumulate](T x)->void{ accumulate = MAX(accumulate, x); },dim, for_all, keepdim);
}

template<Algebraic T>
Tensor<T> Tensor<T>::product(dim_type dim, bool for_all, bool keepdim) const {
    T accumulate = 1;
    return this->reduce_op(accumulate, [&accumulate](T x)->void{ accumulate *= x; },dim, for_all, keepdim);
}

template<Algebraic T>
Tensor<T> Tensor<T>::matmul(const Tensor<T> & other) const {
    assert(dims() == 2 && dims() == other.dims() && "for now implemented only for 2d arrays");
    assert(size[1] == other.size[0] && "shapes of 2d arrays don't match");
    Tensor<T> out(Size({size[0], other.size[1]}));
    if (globals::CPU_MULTITHREAD) {
        assert(0);
        std::vector<std::thread> threads;
    } else {
        assert(0);
    }
    return out;
}

template<Algebraic T>
inline T& Tensor<T>::at(len_type index) const {
    assert(index < numel());
    return data[index];
}

template<Algebraic T>
bool Tensor<T>::all() const {
    return std::all_of(data, data + numel(), [](T x)->bool{ return x; });
}

// FIXME can also be implemented with apply_bin
template<Algebraic T>
Tensor<bool> Tensor<T>::operator==(const Tensor<T> & other) const {
    assert(size == other.size);
    Tensor<bool> out(size);
    for (len_type i = 0; i < numel(); ++i)
        out.at(i) = (data[i] == other.data[i]);
    return out;
}

template<Algebraic T>
len_type Tensor<T>::get_num_threads() const {
    return *std::max_element(size.begin(), size.end());
}

// FIXME: complicated as fuck
template<Algebraic T>
Tensor<T> Tensor<T>::reduce_op(T& accumulate, const std::function<void(T)>& reduction, dim_type dim, bool for_all, bool keepdim) const {
    assert(dim < this->dims() && "Dimension out of bounds (reduce_op)");
    if (for_all) {
        std::for_each(data, data + numel(), reduction);
        Tensor<T> out(1);
        out.data[0] = accumulate;
        return out;
    }
    Tensor<T> out(this->size.remove(dim, keepdim));
    auto temp = accumulate;
    auto block_s = numel(dim);
    auto step = block_s / size[dim];
    len_type count = 0;
    if (globals::CPU_MULTITHREAD && 0) {    // FIXME: for now ~20 times worse than sequential option
        std::vector<std::thread> threads(step);
        auto f = [&reduction, &accumulate, &count, this, block_s, &out, temp, step] (len_type block, int i) -> void {
            for (int delta = 0; delta < block_s; delta += step) {
                assert(block + i + delta < numel());    // DEBUG
                reduction(data[block + i + delta]);
            }
            out.data[count] = accumulate;
            accumulate = temp;
            ++count;
        };
        for (len_type block = 0; block < numel(); block += block_s) {
            for (int i = 0; i < step; ++i) {
                threads[i] = std::thread(f, block, i);
            }
            for (auto& th: threads) {
                th.join();
            }
//            threads.clear();
        }
    } else {
        for (len_type block = 0; block < numel(); block += block_s) {
            for (int i = 0; i < step; ++i) {
                for (int delta = 0; delta < block_s; delta += step) {
                    assert(block + i + delta < numel());    // DEBUG
                    reduction(data[block + i + delta]);
                }
                out.data[count] = accumulate;
                accumulate = temp;
                ++count;
            }
        }
    }
    return out;
}

template<Algebraic T>
Tensor<T> Tensor<T>::sum(dim_type dim, bool for_all, bool keepdim) const {
    T accumulate = 0;
    return this->reduce_op(accumulate, [&accumulate](T x)->void{ accumulate += x; },dim, for_all, keepdim);
}

template<Algebraic T>
Tensor<T> Tensor<T>::sigmoid() const {
    return this->un_op([](T& x)->void{ x = 1 / (exp(-x) + 1); });
}

template<Algebraic T>
Tensor<T> Tensor<T>::tanh() const {
    return this->un_op([](T& x)->void{ T temp = exp(2*x); x = (temp - 1) / (temp + 1); });
}

template<Algebraic T>
Tensor<T> Tensor<T>::e() const {
    return this->un_op([](T& x)->void{ x = exp(x); });
}

template<Algebraic T>
Tensor<T> Tensor<T>::relu() const {
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

// FIXME
template<Algebraic T>
inline Tensor<T>& Tensor<T>::un_op_ip(const std::function<void(T &)>& operation) {
    if (globals::CPU_MULTITHREAD) {
        const auto num_th = this->get_num_threads();
        auto step = numel() / num_th;
        std::vector<std::thread> threads(num_th);
        auto f = [this, operation, step] (int off) -> void{ std::for_each(data + off, data + off + step, operation); };
        for (int off = 0, i = 0; off < numel(); off += step, ++i)
            threads[i] = std::thread(f, off);
        for (auto& th: threads)
            th.join();
    } else {
        std::for_each(data, data + numel(), operation);
    }
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
inline Tensor<T>& Tensor<T>::bin_op_ip(const Tensor<T>& other,const std::function<void(T&, T)>& operation) {
    Comparison state = this->compare(other);
    T* out; len_type out_s;
    T* min_t; len_type min_s;
    switch (state) {
        case Comparison::eq: {
            util::apply_bin_ip<T>(data, numel(), other.data, operation);
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
    if (globals::CPU_MULTITHREAD) {
        auto f = [&out, &min_t, min_s, operation](len_type block) -> void {
            for (int i = 0; i < min_s; ++i)
                operation(out[block + i], min_t[i]);
        };
        std::vector<std::thread> threads(out_s / min_s);
        for (len_type block = 0, i = 0; block < out_s; block += min_s, ++i)
            threads[i] = std::thread(f, block);
        for (auto& th: threads )
            th.join();
    }
    else {
        for (len_type block = 0; block < out_s; block += min_s)
            for (int i = 0; i < min_s; i++)
                operation(out[block + i], min_t[i]);
    }
    return *this;
}

template<Algebraic T>
Comparison Tensor<T>::compare(const Tensor<T> & other) const {
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
inline Tensor<T> Tensor<T>::un_op(const std::function<void(T&)>& operation) const {
    Tensor<T> out(*this);
    return out.un_op_ip(operation);
}

// FIXME: implement a lot of helper - function and optimize the process, which is bloated for now
template<Algebraic T>
Tensor<T> Tensor<T>::bin_op(const Tensor<T> & other, const std::function<void(T&, T)>& operation) const {
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

// FIXME: checkme
template<Algebraic T>
Tensor<T> Tensor<T>::operator[](idx_type index) const {
    assert(index < this->size[0]);
    T* p_out = data + index * numel(1);
    return Tensor<T>(p_out, Size(size, 1), false);
}

template<Algebraic T>
inline dim_type Tensor<T>::dims() const {
    return this->size.dims();
}

// FIXME: not cool style
template<Algebraic T>
Tensor<T>::Tensor(T* new_data, Size&& new_shape, bool is_owner_)
    : size{new_shape}, is_owner{is_owner_} {
    if (is_owner) {
        data = new T[numel()];
        std::copy(new_data, new_data + numel(), data);
    } else {
        data = new_data;
    }
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
        if (this->size == other.size)   // FIXME: if numels match, also no need for realloc
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
    this->size = std::move(other.size);     // nulls other's size
    other.data = nullptr;
	return *this;
}

#if 0
template<Algebraic T>
inline Tensor<T> Tensor<T>::operator[](std::initializer_list<len_type> indexes) const {
	assert(!is_scalar() && "can't subscript scalar tensor");
    len_type arg_size = indexes.size();
    assert(size > indexes && "Not compatible sizes");   // checks for size differences
    T* p_out = this->data + this->getOffset(indexes);
	return Tensor<T>(p_out, Size(this->size, arg_size), false);
}
#endif

template<Algebraic T>
inline Tensor<T> Tensor<T>::operator[](std::initializer_list<len_type> indexes) const {
    assert(!is_scalar() && "can't subscript scalar tensor");
    assert(size > indexes && "out of bounds");
    len_type offset = 0;
    len_type i = 1;
    std::for_each(indexes.begin(), indexes.end(),
                  [&offset, &i, this](len_type x)->void { offset += numel(i) * x; ++i; });
    T* out = data + offset;
    return Tensor<T>(out, Size(size, indexes.size()), false);
}

template<Algebraic T>
inline Size Tensor<T>::shape() const {
	return this->size;
}

template<Algebraic T>
inline len_type Tensor<T>::numel(dim_type start_dim) const {
	assert(start_dim <= this->dims() && "dim number is too high (numel)");
	len_type prod = 1;
	for (len_type i = start_dim; i < this->dims(); ++i) prod *= this->size[i];
	return prod;
}

