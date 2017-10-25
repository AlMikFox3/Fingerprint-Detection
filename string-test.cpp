#include <iostream>
#include <string>
#include <sstream>

namespace patch
{
    template < typename T > std::string to_string( const T& n )
    {
        std::ostringstream stm ;
        stm << n ;
        return stm.str() ;
    }
}
using namespace std;
int main()
{
	string s = patch::to_string(1234);
	s = s + "aaa";
	cout<<s;
	for(int i=1; i<=9; i++){
    for(int j=1; j<=8; j++){
    string dir = "./data/fingerprints/10";
    dir = dir + patch::to_string(i);
    dir = dir + "_";
    dir = dir + patch::to_string(i);
    dir = dir + ".tif";
    cout<<dir;
}}
	return 0;
}
