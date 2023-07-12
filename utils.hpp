#pragma once

#include <cstdint>
#include <iostream> // bad idea btw, FIXME
#include <ostream>
#include <cassert>
#include <initializer_list>
#include <atomic>
#include <concepts>
#include <algorithm>
#include <functional>
#include <utility>
#include <cmath>
#include <limits>

#define MIN(x, y) ((x<y)?x:y)
//#define MAX(x, y) ((x<y)?y:x)

typedef int      idx_type;
//typedef uint32_t uidx_type;
typedef uint16_t count_type;
typedef size_t   len_type;
typedef uint8_t  dim_type;

template<class T>
concept Comparable =
        requires(T self, T other) {
            { self == other } -> std::same_as<bool>;
            { self > other }  -> std::same_as<bool>;
            { self < other }  -> std::same_as<bool>;
        };

template<class T>
concept Algebraic =
        requires(T self, T other) {
            // should have numeric limit infinity
            { -self }        -> std::convertible_to<T>;
            { self }         -> std::convertible_to<bool>; // NOTE: might be a bad idea
            { self == 0 }    -> std::same_as<bool>;        // for handling division by zero
            { self + other } -> std::convertible_to<T>;
            { self - other } -> std::convertible_to<T>;
            { self * other } -> std::convertible_to<T>;
            { self / other } -> std::convertible_to<T>;
        } && Comparable<T>;


enum class Comparison {
    lt,     // less than
    le,     // less or equal than
    gt,     // greater than
    ge,     // greater or equal then
    eq,     // equal
    ne,     // not equal
};

enum class State {
    True,
    False,
    None,
};

namespace util {
    template<typename T>
    inline void apply_ip(T *mutated, std::size_t size, T *other, std::function<void(T &, T)> bin_op) {
        for (std::size_t i = 0; i < size; i++)
            bin_op(mutated[i], other[i]);
    }
}
template<Algebraic T>
class Tensor;

namespace return_types {
    template<Algebraic T>
    class max {
        Tensor<T> indices_;
        Tensor<T> values_;
    public:
        max() = default;
        max(Tensor<T>&& ind, Tensor<T>&& val)
            :indices_{ind}, values_{val} {}
        void operator=(const max&) = delete;
        void operator=(max&&)      = delete;
        Tensor<T> indices() const { return indices_; }
        Tensor<T> values() const { return values_; }
        ~max() = default;
    };
}

// NOTE: data in Tensor can be replaced with this wrapper for simplicity(or no)
template<typename T>
class PtrWrapper {
    T* data = nullptr;
public:
    PtrWrapper() = default;
    explicit PtrWrapper(len_type);
    PtrWrapper(const PtrWrapper& other);
    PtrWrapper(PtrWrapper&& other);
    PtrWrapper& operator=(PtrWrapper&& other);
    PtrWrapper& operator=(const PtrWrapper& other);
    ~PtrWrapper();
};

// NOTE: container which has this iterator must have the following methods:
// begin, end - return iterator, data - returns raw pointer, size - returns size of array
template<typename  Elem>
class ContiguousIterator {
    Elem* ptr = nullptr;
public:
    using value_type = Elem;
    using element_type = Elem;
    using difference_type = std::ptrdiff_t;
    using contiguous_iterator_category = std::contiguous_iterator_tag;
public:
    ContiguousIterator() = default;
    ContiguousIterator(Elem* new_ptr)
            : ptr{new_ptr} {}

    Elem* operator->() const {
        return ptr;
    }

    Elem& operator*() const {
        return *ptr;
    }

    ContiguousIterator& operator++() {
        ptr++;
        return *this;
    }

    ContiguousIterator operator++(int) {
        ContiguousIterator temp = *this;
        ++*this;
        return temp;
    }

    ContiguousIterator& operator--() {
        ptr--;
        return *this;
    }

    ContiguousIterator operator--(int) {
        ContiguousIterator temp = *this;
        --*this;
        return temp;
    }

    ContiguousIterator operator+(difference_type offset) const {
        return ContiguousIterator(ptr + offset);
    }

    friend ContiguousIterator operator+(difference_type offset, const ContiguousIterator& object) {
        return ContiguousIterator(object.ptr + offset);
    }

    ContiguousIterator operator-(difference_type offset) const {
        return ContiguousIterator(ptr - offset);
    }

    difference_type operator-(const ContiguousIterator& other) const {
        return ptr - other.ptr;
    }

    ContiguousIterator& operator+=(difference_type offset) {
        ptr += offset;
        return *this;
    }

    ContiguousIterator& operator-=(difference_type offset) {
        ptr -= offset;
        return *this;
    }

    Elem& operator[](difference_type index) const {
        return ptr[index];
    }

    auto operator<=>(const ContiguousIterator &) const = default;
};













