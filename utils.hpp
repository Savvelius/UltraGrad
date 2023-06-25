#pragma once
#include <cstdint>
#include <iostream> // bad idea btw, FIXME
#include <ostream>
#include <cassert>
#include <initializer_list>

#define MIN(x, y) ((x<y)?x:y)
#define MAX(x, y) ((x<y)?y:x)

typedef int     idx_type;
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

// FIXME if RangeIterator becomes template class, replace this with definiton
class RangeIterator;

// FIXME make me usable pls
#pragma pack(1)
class Range{
private:
    struct{
        len_type start;
        len_type stop;
        len_type step;
    } range_info;
    len_type length = 0;
    len_type* data = nullptr;   // FIXME may be useless
public:
    Range(len_type start_, len_type stop = 0, len_type step = 0) = delete;
    len_type* unfold() = delete;        // FIXME maybe useless
    RangeIterator begin() = delete;     // range iteration
    RangeIterator end() = delete;       // range iteration
    ~Range() = delete;
};

// FIXME read something on for(:) loops to implement this class
// FIXME maybe make it a template Iterator class to use it somewhere else than Range class
class RangeIterator {
private:
    // FIXME pointer of some type should be here with some kind of counter probably
public:
    RangeIterator& operator ++() = delete;        // ++it
    RangeIterator operator ++(int) = delete;     // it++
    RangeIterator& operator --() = delete;        // --it
    RangeIterator operator --(int) = delete;     // it--
    bool operator != (const RangeIterator&) = delete;   // for != for loop comparison
};














