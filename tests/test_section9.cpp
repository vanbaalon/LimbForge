#include "section9_cases.hpp"
int main(){try{Engine e;BatchedLinalg la(e);products<352>(e,la);normal<352>(e,la);real_products<352>(e,la);bridge<224>(e,la);polynomial<352>(e,la);polynomial<704>(e,la);policy();workspace_lifetimes();std::cout<<"Section 9 light smoke passed (not a full audit).\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
