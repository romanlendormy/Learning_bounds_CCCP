#include <gecode/int.hh>
#include <gecode/minimodel.hh>
#include <gecode/search.hh>
#include <iostream>

using namespace Gecode;

class NQueens : public Space {
protected:
    IntVarArray q;
public:
    NQueens(int n) : q(*this, n, 0, n-1) {
        distinct(*this, q);
        for (int i = 0; i < n; i++)
            for (int j = i+1; j < n; j++) {
                rel(*this, q[i] - q[j] != i - j);
                rel(*this, q[j] - q[i] != i - j);
            }
        branch(*this, q, INT_VAR_SIZE_MIN(), INT_VAL_MIN());
    }
    NQueens(NQueens& s) : Space(s) { q.update(*this, s.q); }
    virtual Space* copy() { return new NQueens(*this); }
    void print() const {
        std::cout << "Solution : ";
        for (int i = 0; i < q.size(); i++)
            std::cout << q[i].val() << " ";
        std::cout << std::endl;
    }
};

int main() {
    NQueens* m = new NQueens(8);
    DFS<NQueens> e(m);
    delete m;
    if (NQueens* s = e.next()) {
        std::cout << "Gecode OK !" << std::endl;
        s->print();
        delete s;
    } else {
        std::cout << "Aucune solution trouvee." << std::endl;
    }
    return 0;
}
