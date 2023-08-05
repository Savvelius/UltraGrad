#include "utils.hpp"

bool globals::CPU_MULTITHREAD = true;
bool globals::EXPERIMENTAL    = false;

Comparison operator ! (Comparison cmp) {
    using comp = Comparison;
    switch (cmp) {
        case comp::lt:
            return comp::gt;
        case comp::le:
            return comp::ge;
        case comp::gt:
            return comp::lt;
        case comp::ge:
            return comp::le;
        case comp::eq:
            return comp::ne;
        case comp::ne:
            return comp::eq;
    }
    return cmp;
}

// NOTE: it's a legal Comparison doesn't have bool() operator
Comparison operator && (Comparison cmp, bool val) {
    if (val)
        return cmp;
    return !cmp;
}

std::ostream& operator << (std::ostream& out, Comparison cmp) {
    out << "Comparison(";
    switch (cmp) {
        case Comparison::eq :
            out << "eq";
            break;
        case Comparison::ne :
            out << "ne";
            break;
        case Comparison::lt :
            out << "lt";
            break;
        case Comparison::gt :
            out << "gt";
            break;
        case Comparison::le :
            out << "le";
            break;
        case Comparison::ge :
            out << "ge";
            break;
    }
    out << ')' << std::endl;
    return out;
}

//Comparison operator && (bool val, Comparison cmp) {
//    if (val)
//        return cmp;
//    return !cmp;
//}
