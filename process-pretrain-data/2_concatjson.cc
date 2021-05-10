#include <fstream>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <omp.h>

using namespace std;

void count_words(const string &data, unordered_map<string, int> &dict) {
    size_t i = 0;
    while (true) {
        size_t j = data.find_first_of('"', i);
        if (j == string::npos) {
            break;
        }
        size_t k = data.find_first_of('"', j + 1);
        const string &token = data.substr(j + 1, k - j - 1);
        ++dict[token];
        i = k + 1;
    }
}

void process_file(string in_dir, string out_dir, int num_options, int id, unordered_map<string, int> &dict) {
    vector<unique_ptr<ifstream>> inst_files;
    vector<unique_ptr<ifstream>> state_files;
    vector<unique_ptr<ifstream>> pos_files;
    vector<unique_ptr<ofstream>> inst_out_files;
    vector<unique_ptr<ofstream>> state_out_files;
    vector<unique_ptr<ofstream>> pos_out_files;
    for (int i = 0; i < num_options; ++i) {
        string inst_filename = in_dir + "/" + to_string(i) + "/insts." + to_string(id) + ".json";
        string state_filename = in_dir + "/" + to_string(i) + "/states." + to_string(id) + ".json";
        string pos_filename = in_dir + "/" + to_string(i) + "/pos." + to_string(id) + ".json";
        unique_ptr<ifstream> inst_file(new ifstream(inst_filename));
        unique_ptr<ifstream> state_file(new ifstream(state_filename));
        unique_ptr<ifstream> pos_file(new ifstream(pos_filename));
        if (!(*inst_file) || !(*state_file) || !(*pos_file)) {
            std::cerr << std::strerror(errno) << std::endl;
            return;
        }
        inst_files.emplace_back(move(inst_file));
        state_files.emplace_back(move(state_file));
        pos_files.emplace_back(move(pos_file));
    }
    for (int i = 0; i < num_options; ++i) {
        string option_out_dir = out_dir + "/" + to_string(i);
        string inst_out_filename = option_out_dir + "/insts." + to_string(id) + ".json";
        string state_out_filename = option_out_dir + "/states." + to_string(id) + ".json";
        string pos_out_filename = option_out_dir + "/pos." + to_string(id) + ".json";
        inst_out_files.emplace_back(new ofstream(inst_out_filename));
        state_out_files.emplace_back(new ofstream(state_out_filename));
        pos_out_files.emplace_back(new ofstream(pos_out_filename));
    }
    vector<unordered_map<string, string>> inst_lines(num_options);
    vector<unordered_map<string, string>> state_lines(num_options);
    vector<unordered_map<string, string>> pos_lines(num_options);
    vector<unordered_set<string>> names(num_options);
    vector<string> common_names;
    for (int i = 0; i < num_options; ++i) {
        string name, line;
        while (getline(*inst_files[i], name)) {
            getline(*inst_files[i], line);
            inst_lines[i][name] = line;
            names[i].insert(name);
        }
        while (getline(*state_files[i], name)) {
            getline(*state_files[i], line);
            state_lines[i][name] = line;
        }
        while (getline(*pos_files[i], name)) {
            getline(*pos_files[i], line);
            pos_lines[i][name] = line;
        }
    }
    for (const string &s : names[0]) {
        bool in_all_options = true;
        for (int i = 1; i < num_options; ++i) {
            if (!names[i].count(s)) {
                in_all_options = false;
                break;
            }
        }
        if (in_all_options) {
            common_names.push_back(s);
        }
    }
    for (int i = 0; i < num_options; ++i) {
        for (const string &s : common_names) {
            (*inst_out_files[i]) << inst_lines[i][s] << endl;
            const string &state = state_lines[i][s];
			count_words(state, dict);
            (*state_out_files[i]) << state << endl;
            (*pos_out_files[i]) << pos_lines[i][s] << endl;
        }
    }

}

void merge_dict(unordered_map<string, int> &dict, vector<unordered_map<string, int>> &dicts) {
    for (auto &&d : dicts) {
        for (auto &&it : d) {
            dict[it.first] += it.second;
        }
    }
}

void merge_file(const vector<int> &ids, const vector<string> &tmp_dirs, const string &out_dir, int num_options) {
    vector<unique_ptr<ofstream>> inst_out_files;
    vector<unique_ptr<ofstream>> state_out_files;
    vector<unique_ptr<ofstream>> pos_out_files;
    for (int i = 0; i < num_options; ++i) {
        string option_out_dir = out_dir + "/" + to_string(i);
        mkdir(option_out_dir.c_str(), 0755);
        string inst_out_filename = option_out_dir + "/insts.json";
        string state_out_filename = option_out_dir + "/states.json";
        string pos_out_filename = option_out_dir + "/pos.json";
        inst_out_files.emplace_back(new ofstream(inst_out_filename));
        state_out_files.emplace_back(new ofstream(state_out_filename));
        pos_out_files.emplace_back(new ofstream(pos_out_filename));
    }
    for (int j = 0; j < num_options; ++j) {
        //cout << ids.size() << endl;
        for (size_t i = 0; i < ids.size(); ++i) {
            int id = ids[i];
            const string &tmp_dir = tmp_dirs[i];
            string inst_filename = tmp_dir + "/" + to_string(j) + "/" + "insts." + to_string(id) + ".json";
            string state_filename = tmp_dir + "/" + to_string(j) + "/" + "states." + to_string(id) + ".json";
            string pos_filename = tmp_dir + "/" + to_string(j) + "/" + "pos." + to_string(id) + ".json";
            ifstream inst_file(inst_filename);
            ifstream state_file(state_filename);
            ifstream pos_file(pos_filename);
            if (!inst_file || !state_file || !pos_file) {
                continue;
            }
            string line;
            while (getline(inst_file, line)) {
                (*inst_out_files[j]) << line << endl;
            }
            while (getline(state_file, line)) {
                (*state_out_files[j]) << line << endl;
            }
            while (getline(pos_file, line)) {
                (*pos_out_files[j]) << line << endl;
            }
            //(*inst_out_files[j]) << inst_file.rdbuf();
            //(*state_out_files[j]) << state_file.rdbuf();
            //(*pos_out_files[j]) << pos_file.rdbuf();
        }
    }
}

int get_files_num(const string &in_dir, int num_options) {
    int min_id = -1;
    for (int i = 0; i < num_options; ++i) {
        int max_id = 0;
        const string &option_dir = in_dir + "/" + to_string(i);
        DIR *d = opendir(option_dir.c_str());
        if (!d) {
            return 0;
        }
        for (struct dirent *dir = readdir(d); dir; dir = readdir(d)) {
            string name = dir->d_name;
            if (name.find("pos") != string::npos) {
                int id = stoi(name.substr(4, name.length() - 9));
                if (id > max_id) {
                    max_id = id;
                }
            }
        }
        closedir(d);
        if (min_id == -1 || max_id < min_id) {
            min_id = max_id;
        }
    }
    return min_id + 1;
}

int main(int argc, char **argv) {
    string out_dir = argv[1];
    string tmp_dir = argv[2];
    string dict_filename = argv[3];
    int num_options = atoi(argv[4]);
    int num_threads = atoi(argv[5]);
    vector<string> in_dirs;
    vector<double> rates;
    int num_dirs = (argc - 6) / 2;
    for (int i = 0; i < num_dirs; ++i) {
        in_dirs.push_back(argv[i + 6]);
        rates.push_back(atof(argv[i + 6 + num_dirs]));
        mkdir((tmp_dir + "/" + to_string(i)).c_str(), 0755);
        for (int j = 0; j < num_options; ++j) {
            const string &opt_dir = tmp_dir + "/" + to_string(i) + "/" + to_string(j);
            mkdir(opt_dir.c_str(), 0755);
        }
    }
    for (int i = 0; i < num_options; ++i) {
        mkdir((out_dir + "/" + to_string(i)).c_str(), 0755);
    }
    vector<int> ids;
    vector<string> dirs, tmp_dirs;
    vector<unordered_map<string, int>> dicts(num_threads);
    unordered_map<string, int> total_dict;
    for (int i = 0; i < num_dirs; ++i) {
        default_random_engine gen;
        gen.seed(0);
        bernoulli_distribution dist(rates[i]);
        int num_files = get_files_num(in_dirs[i], num_options);
        for (int j = 0; j < num_files; ++j) {
            if (dist(gen)) {
                dirs.push_back(in_dirs[i]);
                tmp_dirs.push_back(tmp_dir + "/" + to_string(i));
                ids.push_back(j);
            }
        }
    }
    #pragma omp parallel for num_threads(num_threads)
    for (size_t i = 0; i < ids.size(); ++i) {
        int id = ids[i];
        string in_dir = dirs[i];
        string out_dir = tmp_dirs[i];
        unordered_map<string, int> &dict = dicts[omp_get_thread_num()];
        process_file(in_dir, out_dir, num_options, id, dict);
    }
    merge_dict(total_dict, dicts);
    merge_file(ids, tmp_dirs, out_dir, num_options);
    ofstream dict_file(dict_filename);
    for (auto &&it : total_dict) {
        dict_file << it.first << " " << it.second << endl;
    }
    return 0;
}