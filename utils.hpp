#pragma once
#include <cstdint>
#include <iostream> // bad idea btw, FIXME
#include <ostream>
#include <cassert>
#include <initializer_list>

#define MIN(x, y) ((x<y)?x:y)
#define MAX(x, y) ((x<y)?y:x)

typedef int     idx_type;
typedef int64_t T;
typedef size_t  len_type;
typedef uint8_t dim_type;

#pragma pack(1)
template<typename T>
class SharedPtr {
private:
    T* data = nullptr;
    uint16_t count = 0;
public:
    SharedPtr() = default;
    explicit SharedPtr(T *);
    explicit SharedPtr(size_t);     // NOTE: memory is allocated only here
    SharedPtr(SharedPtr<T>&);
    SharedPtr(const SharedPtr<T>&) = delete;
    SharedPtr(SharedPtr<T>&&) noexcept;

    SharedPtr<T>& operator=(SharedPtr<T>&);
    SharedPtr<T>& operator=(const SharedPtr<T>&) = delete;
    SharedPtr<T>& operator=(SharedPtr<T>&&);

    T& operator*() const;
    T& operator[](size_t) const;

    ~SharedPtr();
};

template<typename T>
SharedPtr<T> &SharedPtr<T>::operator=(SharedPtr<T> && other) {
    if (this == &other)
        return *this;
    this->data = other.data;
    this->count = other.count;

    other.data = nullptr;
    other.count = 0;
    return *this;
}

template<typename T>
SharedPtr<T> &SharedPtr<T>::operator=(SharedPtr<T> & other) {
    if (this == &other)
        return *this;
    this->data = other.data;
    other.count ++;
}

template<typename T>
SharedPtr<T>::SharedPtr(SharedPtr<T> && other) noexcept {
    this->data = other.data;
    this->count = other.count;

    other.data = nullptr;
    other.count = 0;
}

template<typename T>
SharedPtr<T>::SharedPtr(SharedPtr<T> & other) {
    this->data = other.data;
    other.count ++;     // probably
}

template<typename T>
SharedPtr<T>::SharedPtr(T *data_): data(data_) {
}

template<typename T>
inline T &SharedPtr<T>::operator[](size_t index) const {
    return *(data + index);
}

template<typename T>
inline SharedPtr<T>::SharedPtr(size_t size) {
    this->data = new T[size];
}

template<typename T>
inline T& SharedPtr<T>::operator*() const {
    return *data;
}

template<typename T>
SharedPtr<T>::~SharedPtr() {
    if (this->count == 0){
        delete[] this->data;
    }
}



#pragma pack(1)
class Size{
private:
    len_type* data = nullptr;
    dim_type length = 0;
public:
    Size() = default;
    Size(std::initializer_list<len_type>);
    Size(const Size&, dim_type);    // creates a copy of Size from given dim
    Size(const Size&);
    Size(Size&&) noexcept;

    Size& operator =(const Size&);
    Size& operator =(Size&&) noexcept;
    Size& operator =(std::initializer_list<len_type>);

    bool operator ==(std::initializer_list<len_type>) const;
    bool operator ==(const Size&) const;
    bool operator <(const Size&) const;
    bool operator >(const Size&) const;
    bool operator <(std::initializer_list<len_type>) const;
    bool operator >(std::initializer_list<len_type>) const;

    len_type operator[](dim_type) const;
    dim_type dims() const;
    len_type numel(dim_type start_dim = 0) const;
    dim_type index(len_type) const;
    dim_type count(len_type) const;

    ~Size();

    friend std::ostream& operator <<(std::ostream&, const Size&);
    template<class T>
    friend class Tensor;    // FIXME maybe wrong to do this
};

// FIXME read something on for(:) loops to implement this class
// FIXME maybe make it a template Iterator class to use it somewhere else than Range class
template<typename Container, typename Element>
class Iterator__ {
private:
    // FIXME pointer of some type should be here with some kind of counter probably
    Container iterable;
    Element* current = nullptr;
public:
    using iterator = Iterator__<Container, Element>;

    iterator& operator ++() = delete;        // ++it
    iterator operator ++(int) = delete;     // it++
    iterator& operator --() = delete;        // --it
    iterator operator --(int) = delete;     // it--
    bool operator != (const iterator&) = delete;   // for != for loop comparison
};

// FIXME i am usable
#pragma pack(1)
template<typename T>
class Range {
private:
    struct{
        T start = 0;
        T stop  = 0;
        T step  = 0;
    } range_info;
    struct Iterator {   // FIXME may be implemented as another double template class
        T current = 0;
        T step = 0;
        Iterator() = default;
        Iterator(T start, T step);
        Iterator operator ++(int) &;
        Iterator& operator ++();     // actually this gets called in for(:) loop
        T operator*();
        bool operator != (const Iterator&);
    };
public:
    Range(T start_, T stop_ = 0, T step_ = 0);
    len_type* unfold() = delete;        // FIXME maybe useless
    Iterator begin();     // range iteration
    Iterator end();       // range iteration
    ~Range() = default;

    friend Iterator;
};


template<typename T>
typename Range<T>::Iterator Range<T>::Iterator::operator++(int) & {
    auto temp = *this;
    this->current += this->step;
    return temp;
}

template<typename T>
bool Range<T>::Iterator::operator!=(const Range<T>::Iterator & other) {
    return this->current < other.current;
}

template<typename T>
Range<T>::Iterator::Iterator(T start, T step) {
    this->current = start;
    this->step = step;
}

template<typename T>
typename Range<T>::Iterator& Range<T>::Iterator::operator++() {
    this->current += this->step;
    return *this;
}

template<typename T>
T Range<T>::Iterator::operator*() {
    return this->current;
}

template<typename T>
Range<T>::Range(T start_, T stop_, T step_) {
    if (start_ > stop_)
        assert(step_ < 0);
    else if (start_ != stop_)
        assert(step_ > 0);
    this->range_info = {start_, stop_, step_};
}

template<typename T>
typename Range<T>::Iterator Range<T>::begin() {
    return Range::Iterator(this->range_info.start, this->range_info.step);
}
template<typename T>
typename Range<T>::Iterator Range<T>::end() {
    return Range::Iterator(this->range_info.stop, 0);
}















