#include"myRNA.h"

int main(){
    string str;
    cin>>str;
    RNA r;
    r.load(str);
    r.show();
    r.showbin();
    r.showinverse();
    return 0;
}