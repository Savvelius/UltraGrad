#include "utils.hpp"

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

// NOTE: it's legal while it doesn't have bool() operator
Comparison operator && (Comparison cmp, bool val) {
    if (val)
        return cmp;
    return !cmp;
}

//Comparison operator && (bool val, Comparison cmp) {
//    if (val)
//        return cmp;
//    return !cmp;
//}
