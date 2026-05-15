#include <xgboost/c_api.h>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <map>
#include <iostream>
#include <iomanip>
#include "json.hpp" 

using json = nlohmann::json;

const double LOW_THR  = 0.20;
const double HIGH_THR = 0.60;


std::vector<double> parse_power_series(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) out.push_back(std::stod(item));
    }
    return out;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1) return v[n/2];
    return (v[n/2-1] + v[n/2]) * 0.5;
}

std::pair<double,double> get_bounds_from_model(const json& boundary_json, double temperature) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << temperature;
    std::string key = ss.str();

    auto it = boundary_json.find(key);
    if (it == boundary_json.end()) return {0.0,0.0};

    auto arr = *it;

    double bmin = std::stod(arr[0].get<std::string>());
    double bmax = std::stod(arr[1].get<std::string>());

    return {bmin, bmax};
}

extern "C" __declspec(dllexport)
int run_trial_class(
    const char* model_path,
    const char* series_str,
    double temperature,
    char* out_features,   
    int out_size          
) {

    std::string full_model_path = std::string("models/") + model_path;

    std::vector<double> power = parse_power_series(series_str);
    if (power.empty()) return -1;

    DMatrixHandle dmat_tmp;
    if (XGDMatrixCreateFromMat(nullptr, 0, 0, NAN, &dmat_tmp) != 0) return -2;

    BoosterHandle booster;
    if (XGBoosterCreate(&dmat_tmp, 1, &booster) != 0) return -3;
    if (XGBoosterLoadModel(booster, full_model_path.c_str()) != 0) return -4;

    const char* attr_val = nullptr;
    int success = 0;

    if (XGBoosterGetAttr(booster, "boundary_table", &attr_val, &success) != 0 || !success) {
        return -5;
    }

    json boundary_json = json::parse(attr_val);
    auto [bmin, bmax] = get_bounds_from_model(boundary_json, temperature);

     size_t n = power.size();

    double sum = 0.0;
    for (double v : power) sum += v;
    double mean = sum / n;

    double sq = 0.0;
    for (double v : power)
        sq += (v - mean) * (v - mean);

    double stdv = std::sqrt(sq / n);

    double pmin = *std::min_element(power.begin(), power.end());
    double pmax = *std::max_element(power.begin(), power.end());

    double width = (bmax - bmin) + 1e-6;

    double under_sum = 0.0, over_sum = 0.0;
    double under_max = 0.0, over_max = 0.0;

    int cnt_under = 0;
    int cnt_over = 0;

    double max_jump_raw = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double u = std::max(bmin - power[i], 0.0) / width;
        double o = std::max(power[i] - bmax, 0.0) / width;

        under_sum += u;
        over_sum  += o;

        if (u > under_max) under_max = u;
        if (o > over_max)  over_max = o;

        if (power[i] < bmin) cnt_under++;
        if (power[i] > bmax) cnt_over++;

        if (i > 0) {
            double jump = std::abs(power[i] - power[i - 1]);
            if (jump > max_jump_raw) max_jump_raw = jump;
        }
    }

    double under_mean = under_sum / n;
    double over_mean  = over_sum / n;

    double under_frac = (double)cnt_under / n;
    double over_frac  = (double)cnt_over / n;

    double max_jump = max_jump_raw / width;

    double slope = 0.0;

    if (n > 1) {
        double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;

        for (size_t i = 0; i < n; ++i) {
            double x = (double)i;
            double y = power[i];

            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        double denom = n * sum_x2 - sum_x * sum_x;

        if (std::abs(denom) > 1e-12) {
            slope = (n * sum_xy - sum_x * sum_y) / denom;
            double intercept = (sum_y - slope * sum_x) / n;

            double err = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double trend = slope * i + intercept;
                err += std::abs(power[i] - trend);
            }

        }
    }

    double trend_slope = slope / width;


    double mean_pos = (mean - bmin) / width;
    double margin_to_min = (pmin - bmin) / width;
    double margin_to_max = (bmax - pmax) / width;

    double power_cv = stdv / (mean + 1e-6);

    double data[12] = {
        under_mean,
        under_max,
        over_mean,
        over_max,
        under_frac,
        over_frac,
        mean_pos,
        margin_to_min,
        margin_to_max,
        max_jump,
        trend_slope,
        power_cv
    };

    const char* feature_names[12] = {
        "under_mean",
        "under_max",
        "over_mean",
        "over_max",
        "under_frac",
        "over_frac",
        "mean_pos",
        "margin_to_min",
        "margin_to_max",
        "max_jump",
        "trend_slope",
        "power_cv"
    };
    // for (int i = 0; i < 12; ++i)
    //     std::cout << feature_names[i] << " = " << data[i] << std::endl;


    DMatrixHandle dmat;
    float data_f[12];
    for(int i=0;i<12;i++) data_f[i] = (float)data[i];

    if (XGDMatrixCreateFromMat(data_f,1,12,NAN,&dmat)!=0) return -6;

    bst_ulong out_len;
    const float* out;
    if (XGBoosterPredict(booster,dmat,0,0,0,&out_len,&out)!=0) return -7;

    double prob = out[0];

    // -------- SHAP --------

    const char* json_config = R"({
        "type": 2,
        "training": false,
        "iteration_begin": 0,
        "iteration_end": 0,
        "strict_shape": true
    })";

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

    // size
    bst_ulong total = 1;
    for (bst_ulong i = 0; i < out_dim; ++i)
        total *= out_shape[i];

    float bias = contribs[total - 1];

    // std::cout << "\n SHAP \n";
    // for (int i = 0; i < 12; ++i) {
    //     std::cout 
    //         << feature_names[i] 
    //         << " (" << data[i] << ") => " 
    //         << contribs[i] 
    //         << std::endl;
    // }

    // std::cout << "bias = " << bias << std::endl;

    double logit = bias;
    for (int i = 0; i < 12; ++i)
        logit += contribs[i];

    double prob_recalc = 1.0 / (1.0 + std::exp(-logit));

    // все фичи

    std::vector<std::pair<std::string, float>> feats;
    for (int i = 0; i < 12; ++i)
        feats.push_back({feature_names[i], contribs[i]});

    std::sort(feats.begin(), feats.end(),
        [](const auto& a, const auto& b){
            return std::abs(a.second) > std::abs(b.second);
        });

    std::stringstream ss;
    ss << "prob:" << std::fixed << std::setprecision(2) << prob << ", ";
    for (int i = 0; i < 5; ++i) {
        ss << feats[i].first << ":" << feats[i].second;
        if (i != 4) ss << ", ";
    }

    std::string features_str = ss.str();
    strncpy(out_features, features_str.c_str(), out_size - 1);
    out_features[out_size - 1] = '\0';

    XGDMatrixFree(dmat);
    XGBoosterFree(booster);
    XGDMatrixFree(dmat_tmp);

    return prob <= LOW_THR ? 2 : (prob >= HIGH_THR ? 1 : 3);
}