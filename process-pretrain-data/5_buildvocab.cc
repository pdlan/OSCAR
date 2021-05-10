#include <fstream>
#include <unordered_map>
#include <thread>
#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;

bool compare(const pair<string, int> &p1, const pair<string, int> &p2) {
    return p1.second > p2.second;
}

int main(int argc, char **argv) {
    if (argc != 5) {
        cout << "./5_build_vocab <stage4 in> <num options> <inst dict> <state dict>\n";
        return 0;
    }
    string in_dir = argv[1];
    int num_options = atoi(argv[2]);
    string inst_dict_filename = argv[3];
    string state_dict_filename = argv[4];
    unordered_map<string, int> inst_dict;
    unordered_map<string, int> state_dict;
    for (int i = 0; i < num_options; ++i) {
        ifstream ifs1(in_dir + "/" + to_string(i) + "/insts_train.txt");
        ifstream ifs2(in_dir + "/" + to_string(i) + "/states_train.txt");
        string token;
        while (ifs1 >> token) {
            ++inst_dict[token];
        }
        while (ifs2 >> token) {
            ++state_dict[token];
        }
        cout << "Finish counting words from " << i << endl;
    }
    ofstream ofs1(inst_dict_filename);
    ofstream ofs2(state_dict_filename);
    vector<pair<string, int>> inst_dict_sorted;
    vector<pair<string, int>> state_dict_sorted;
    inst_dict.erase("<unk>");
    inst_dict.erase("</s>");
    state_dict.erase("<unk>");
    state_dict.erase("</s>");
    for (auto &&it : inst_dict) {
        inst_dict_sorted.push_back(it);
    }
    for (auto &&it : state_dict) {
        state_dict_sorted.push_back(it);
    }
    sort(inst_dict_sorted.begin(), inst_dict_sorted.end(), compare);
    sort(state_dict_sorted.begin(), state_dict_sorted.end(), compare);
    for (auto &&it : inst_dict_sorted) {
        ofs1 << it.first << " " << it.second << endl;
    }
    for (auto &&it : state_dict_sorted) {
        ofs2 << it.first << " " << it.second << endl;
    }
    return 0;
}