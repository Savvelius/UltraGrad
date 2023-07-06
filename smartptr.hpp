#pragma once

#include "utils.hpp"

// NOTE: if implement cpu multi-threading, use mutex / atomic as ref_count
template<typename T>
class SharedPtr {
private:
    T* data_ = nullptr;
    struct Resource {
        T *memory             = nullptr;
        count_type *ref_count = nullptr;

        Resource() = default;
        Resource(Resource&);
        Resource(Resource&&);
        bool reserve(len_type);
        Resource& operator=(Resource&);
        Resource& operator=(Resource&&);
        ~Resource();
    } resource;
    void swap(SharedPtr<T>&) = delete;
public:
    SharedPtr() = default;
    explicit SharedPtr(len_type);     // NOTE: memory is allocated only here
    bool reserve(len_type);           // and here if it wasn't already allocated

    SharedPtr(SharedPtr<T>&)           = default;
    SharedPtr(SharedPtr<T>&, len_type);
    SharedPtr(SharedPtr<T>&&) noexcept = default;

    SharedPtr<T>& operator=(SharedPtr<T>&)           = default;
    SharedPtr<T>& operator=(SharedPtr<T>&&) noexcept = default;

    T& operator*() const;
    T& operator[](len_type) const;
    T* data() const;

    ~SharedPtr() = default;
};

template<typename T>
T *SharedPtr<T>::data() const {
    return data_;
}

template<typename T>
SharedPtr<T>::SharedPtr(SharedPtr<T> & other, len_type length) {
    resource = other.resource;
    data_     = other.data_ + length;
}

template<typename T>
typename SharedPtr<T>::Resource &SharedPtr<T>::Resource::operator=(SharedPtr::Resource && other) {
    this->~Resource();
    ref_count = other.ref_count;
    memory    = other.memory;

    other.ref_count = nullptr;
    other.memory    = nullptr;
    return *this;
}

template<typename T>
typename SharedPtr<T>::Resource& SharedPtr<T>::Resource::operator=(SharedPtr::Resource & other) {
    this->~Resource();
    ref_count = other.ref_count;
    memory    = other.memory;

    ++(*ref_count);
    return *this;
}

template<typename T>
SharedPtr<T>::Resource::Resource(SharedPtr::Resource && other) {
    ref_count = other.ref_count;
    memory    = other.memory;

    other.ref_count = nullptr;
    other.memory    = nullptr;
}

template<typename T>
SharedPtr<T>::Resource::Resource(SharedPtr::Resource & other) {
    ref_count = other.ref_count;
    memory    = other.memory;
    ++(*ref_count);
}

template<typename T>
bool SharedPtr<T>::Resource::reserve(len_type length) {
    if (memory)
        return false;

    memory    = new T[length];
    ref_count = new count_type;

    return true;
}

template<typename T>
SharedPtr<T>::Resource::~Resource() {
    if (!memory)    // in case of bullshit
        return;

    if (!--*ref_count){     // if no copies left, delete resources
        delete ref_count;
        delete[] memory;
    }

    ref_count = nullptr;
    memory    = nullptr;
}

template<typename T>
T& SharedPtr<T>::operator[](len_type index) const {
    return this->data_[index];
}

template<typename T>
T& SharedPtr<T>::operator*() const {
    return *(this->data_);
}

template<typename T>
bool SharedPtr<T>::reserve(len_type length) {
    if (this->resource.reserve(length)) {
        this->data_ = resource.memory;
        return true;
    }
    return false;
}

template<typename T>
SharedPtr<T>::SharedPtr(len_type length) {
    resource.reserve(length);
    data_ = resource.memory;
}

