#pragma once
#include "utils.hpp"

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

// FIXME: read about cpp ranges, maybe delete this(or at least not use it in project)
template<typename T>
class Range {
private:
    struct RangeInfo {
        T start = 0;
        T stop  = 0;
        T step  = 0;
    } range_info;
    struct Iterator {   // FIXME may be implemented as another double template class
        T current = 0;
        T step = 0;
        bool forward = true;
        Iterator() = default;
        Iterator(T start, T step, bool is_forward = true);
        Iterator operator ++(int) &; // useless for now
        Iterator& operator ++();     // actually this gets called in for(:) loop
        T operator*();
        bool operator != (const Iterator&);
    };
    bool forward = true;        // THINK: delete this to gain 4b and not lose performance???
public:
    Range(T start_, T stop_ = 0, T step_ = 0);
    RangeInfo get_info();
    idx_type size();
    Iterator begin() const;     // range iteration
    Iterator end()   const;       // range iteration
    ~Range() = default;

    template<typename U>
    friend std::ostream& operator <<(std::ostream&, const Range<U>&);
    friend Iterator;
};

template<typename T>
typename Range<T>::RangeInfo Range<T>::get_info() {
    return Range::RangeInfo(this->range_info);
}

template<typename T>
std::ostream &operator<<(std::ostream & out, const Range<T> & obj) {
    out << "Range(";
    for (auto i : obj)
        out << i << ", ";
    out << ')' << std::endl;
    return out;
}

template<typename T>
idx_type Range<T>::size() {
    if (range_info.start == range_info.stop)
        return 0;
    return (idx_type)(abs(this->range_info.stop - this->range_info.start) / abs(this->range_info.step)) + 1;
}


template<typename T>
typename Range<T>::Iterator Range<T>::Iterator::operator++(int) & {
    auto temp = *this;
    this->current += this->step;
    return temp;
}

template<typename T>
bool Range<T>::Iterator::operator!=(const Range<T>::Iterator & other) {
    switch (this->forward) {
        case true:
            return this->current < other.current;
        case false:
            return this->current > other.current;
    }

}

template<typename T>
Range<T>::Iterator::Iterator(T start, T step, bool is_forward) {
    this->current = start;
    this->step = step;
    this->forward = is_forward;
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
    if (start_ > stop_) {
        assert(step_ < 0);
        this->forward = false;
    }
    else if (start_ != stop_)
        assert(step_ > 0);
    this->range_info = {start_, stop_, step_};
}

template<typename T>
typename Range<T>::Iterator Range<T>::begin() const {
    return Range::Iterator(this->range_info.start, this->range_info.step, this->forward);
}
template<typename T>
typename Range<T>::Iterator Range<T>::end() const {
    return Range::Iterator(this->range_info.stop, 0);
}
