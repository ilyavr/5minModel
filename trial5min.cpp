#include <xgboost/c_api.h>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>
#include <iostream>
#include "json.hpp" 

using json = nlohmann::json;

const double LOW_THR  = 0.12;
const double HIGH_THR = 0.95;

//разбивает строку мощности 
std::vector<double> parse_power_series(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stod(item));
    }
    return out;
}
// считает медиану
double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n/2];
    return (v[n/2-1] + v[n/2]) * 0.5;
}

// Функция для получения границ из JSON строки хранящейся в модели
// Она имеет вид:
// {
// "20.0": [bmax, bmin],
// "20.1": [bmax, bmin]
// }
// ключ - температура, значение - максимальная и минимальная граница
std::pair<double,double> get_bounds_from_model(const json& boundary_json, double temperature) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << temperature;
    std::string key = ss.str();

    auto it = boundary_json.find(key);
    if (it == boundary_json.end()) return {0.0,0.0};

    auto arr = *it;

    double bmax = std::stod(arr[0].get<std::string>());
    double bmin = std::stod(arr[1].get<std::string>());

    return {bmax, bmin};
}


// Отдавать аргументы в виде:
// run_trial_class(
//         "0418_model.json",
//         "84.5,85,85.5,86.5",
//         30.7
//     )
// название файла с весами, мощности, температура
// номер модели нужно получать как первые 4 символа баркода
extern "C" __declspec(dllexport)
int run_trial_class(const char* model_path, const char* series_str, double temperature, const char** out_features) {
    static std::string features_str;
    std::string full_model_path = std::string("models/") + model_path;

    std::vector<double> power = parse_power_series(series_str);

    // Нет мощности = -1
    // Не собралась матрица-заглушка для вызова модели бустера = -2
    // Не загрузилась модель booster'a = -3
    // Неправильный путь к весам =-4
    // В весах нет JSON строки температура-границы = -5
    // Не собралась матрица из вычисленных фич = -6
    // Не получилось предсказать годность = -7

    if (power.empty()) return -1;

    DMatrixHandle dmat_tmp;
    if (XGDMatrixCreateFromMat(nullptr, 0, 0, NAN, &dmat_tmp) != 0) return -2;

    BoosterHandle booster;
    if (XGBoosterCreate(&dmat_tmp, 1, &booster) != 0) return -3;
    if (XGBoosterLoadModel(booster, full_model_path.c_str()) != 0) return -4;

    const char* attr_val = nullptr;
    int success = 0;

    if (XGBoosterGetAttr(booster, "boundary_table", &attr_val, &success) != 0 || !success) {
        XGBoosterFree(booster);
        XGDMatrixFree(dmat_tmp);
        return -5;
    }

    
    //Вычисление фич

    //Получение границ из весов
    json boundary_json = json::parse(attr_val);
    auto [bmax, bmin] = get_bounds_from_model(boundary_json, temperature);

    size_t n = power.size();
    double sum = 0.0;
    for (double v : power) sum += v;
    double mean = sum / n;

    double sq = 0.0;
    for (double v : power) sq += (v - mean)*(v - mean);
    double stdv = std::sqrt(sq / n);

    double pmin = *std::min_element(power.begin(), power.end());
    double pmax = *std::max_element(power.begin(), power.end());
    double prange = pmax - pmin;
    double med = median(power);

    double max_jump = 0.0;
    for (size_t i = 1; i < n; ++i) max_jump = std::max(max_jump, std::abs(power[i]-power[i-1]));

    double max_over_max = 0.0, max_under_min = 0.0;

    for (double p : power){
        if (p>bmax) 
            max_over_max= p - bmax;
        if (p > bmin)
            max_under_min = bmin - p;
    }

    int cnt_over_max = 0;
    int cnt_under_min = 0; 
    for (double p : power){
        if (p>bmax) cnt_over_max++;
        if (p<bmin) cnt_under_min++;
    }

    double frac_over_max = (double)cnt_over_max/n;
    double frac_under_min = (double)cnt_under_min/n;

    // Массив с посчитанными фичами
    double data[17] = {
        bmax, bmin,
        (double)n, mean, med, stdv,
        pmin, pmax, prange,
        (double)cnt_over_max, (double)cnt_under_min,
        frac_over_max, frac_under_min,
        max_over_max, max_under_min,
        max_jump,
        temperature
    };


    const char* feature_names[17] = {
        "boundaryPowerMax","boundaryPowerMin","power_len","power_mean","power_median","power_std",
        "power_min","power_max","power_range","cnt_over_max","cnt_under_min",
        "frac_over_max","frac_under_min","max_over_max","max_under_min","max_jump","temperature"
    };

    // for (int i = 0; i < 17; ++i) {
    //     std::cout << feature_names[i] << " = " << data[i] << std::endl;
    // }


    DMatrixHandle dmat;
    float data_f[17];
    for(int i=0;i<17;i++) data_f[i] = (float)data[i];

    XGDMatrixCreateFromMat(data_f, 1, 17, NAN, &dmat);
    if (XGDMatrixCreateFromMat(data_f,1,17,NAN,&dmat)!=0) return -6;

    bst_ulong out_len;
    const float* out;
    if (XGBoosterPredict(booster,dmat,0,0,0,&out_len,&out)!=0) return -7;

    double prob = out[0];
        const char* json_config = R"({
            "type": 2,
            "training": false,
            "iteration_begin": 0, 
            "iteration_end": 0,
            "strict_shape": true 
        })";

        // const char* json_config = R"({
        //     "type": 2,                   shap вклады
        //     "training": false,           
        //     "iteration_begin": 0,         диапазон деревьев
        //     "iteration_end": 0,           если индекс последнего дерева 0 то будет использоваться весь ансамбль
        //     "strict_shape": true          проверка согласованности размерности выходного массива (массива фич)           
        // })";
    const bst_ulong* out_shape;
    bst_ulong out_dim = 0;
    const float* contribs;

if (XGBoosterPredictFromDMatrix(
        booster,
        dmat,
        json_config,
        &out_shape,
        &out_dim,
        &contribs) != 0) return -8;
    // считаем размер массива который возвращает предикт
    bst_ulong contrib_len = 1; 
    for (bst_ulong i = 0; i < out_dim; ++i)
        contrib_len *= out_shape[i];

    // std::cout << "contrib_len = " << contrib_len << std::endl;
    // std::cout << "out_dim = " << out_dim << std::endl;
    std::vector<std::pair<std::string, float>> feats;
    //  фича - вклад
    for (int i = 0; i < 17; ++i) 
        feats.push_back({feature_names[i],contribs[i]});
    
    // сортируем вклад каждой фичи по модулю
    std::sort(feats.begin(),feats.end(),[](const auto& a, const auto& b){return std::abs(a.second) > std::abs(b.second);});
    // std::cout<< " ==========================================================================="<<std::endl;
    // for (const auto& f: feats){
    //     std::cout<<f.first<< " = "<<f.second<<std::endl;
    // }
    // std::cout<< " ==========================================================================="<<std::endl;
    std::stringstream ss;
    for (int i = 0; i < 5 && i < feats.size(); ++i) {
        ss << feats[i].first << ":" << feats[i].second;
        if (i != 4 && i != feats.size()-1) ss << ", ";
    }

    features_str = ss.str();
    * out_features = features_str.c_str();
    // std::cout << "1 = " << ss.str() << std::endl;
    // std::cout << "2 = " << features_str << std::endl;

    XGDMatrixFree(dmat);
    XGBoosterFree(booster);
    XGDMatrixFree(dmat_tmp);
    // std::cout << "prob = " << prob << std::endl;
    return prob <= LOW_THR ? 2 : (prob >= HIGH_THR ? 1 : 3);
}