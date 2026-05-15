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
const double HIGH_THR = 0.80;

// ---------------- utils ----------------

std::vector<double> parse_power_series(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;

    while (std::getline(ss, item, ',')) {
        if (!item.empty())
            out.push_back(std::stod(item));
    }

    return out;
}

std::pair<double,double> get_bounds_from_model(const json& boundary_json, double temperature) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1) << temperature;
    std::string key = ss.str();

    auto it = boundary_json.find(key);
    if (it == boundary_json.end()) return {0.0, 0.0};

    auto arr = *it;
    double bmin = std::stod(arr[0].get<std::string>());
    double bmax = std::stod(arr[1].get<std::string>());

    return {bmin, bmax};
}

// ---------------- MAIN ----------------

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

    if (bmax <= bmin) return -6;

    size_t n = power.size();
    double width = (bmax - bmin) + 1e-6;

    // ---------------- stats ----------------

    double sum = 0.0;
    for (double v : power) sum += v;
    double mean = sum / n;

    double sq = 0.0;
    for (double v : power)
        sq += (v - mean) * (v - mean);

    double stdv = std::sqrt(sq / n);

    double pmin = *std::min_element(power.begin(), power.end());
    double pmax = *std::max_element(power.begin(), power.end());

    // ---------------- under/over ----------------

    double under_sum = 0.0, over_sum = 0.0;
    double under_max = 0.0, over_max = 0.0;
    int cnt_under = 0, cnt_over = 0;

    double max_jump = 0.0;

    for (size_t i = 0; i < n; ++i) {
        double u = std::max(bmin - power[i], 0.0) / width;
        double o = std::max(power[i] - bmax, 0.0) / width;

        under_sum += u;
        over_sum += o;

        under_max = std::max(under_max, u);
        over_max = std::max(over_max, o);

        if (power[i] < bmin) cnt_under++;
        if (power[i] > bmax) cnt_over++;

        if (i > 0)
            max_jump = std::max(max_jump, std::abs(power[i] - power[i-1]));
    }

    double under_mean = under_sum / n;
    double over_mean  = over_sum / n;

    double under_frac = (double)cnt_under / n;
    double over_frac  = (double)cnt_over / n;

    max_jump /= width;

    // ---------------- trend (OLS like np.polyfit) ----------------

    double slope = 0.0;
    double residual = 0.0;

    if (n > 1) {
        double sx = 0, sy = 0, sxy = 0, sx2 = 0;

        for (size_t i = 0; i < n; ++i) {
            double x = (double)i;
            double y = power[i];

            sx += x;
            sy += y;
            sxy += x * y;
            sx2 += x * x;
        }

        double denom = n * sx2 - sx * sx;

        if (std::abs(denom) > 1e-12) {
            slope = (n * sxy - sx * sy) / denom;
            double intercept = (sy - slope * sx) / n;

            double err = 0.0;
            for (size_t i = 0; i < n; ++i) {
                double pred = slope * i + intercept;
                err += std::abs(power[i] - pred);
            }

            residual = (err / n) / width;
        }
    }

    double trend_slope = slope / width;

    // ---------------- final features ----------------

    double mean_pos = (mean - bmin) / width;
    double margin_to_min = (pmin - bmin) / width;
    double margin_to_max = (bmax - pmax) / width;

    double power_cv = stdv / (mean + 1e-6);

    double data[13] = {
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
        residual,
        power_cv
    };

    const char* feature_names[13] = {
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
        "trend_residual",
        "power_cv"
    };

    // ---------------- prediction ----------------

    float data_f[13];
    for (int i = 0; i < 13; i++) data_f[i] = (float)data[i];

    DMatrixHandle dmat;
    if (XGDMatrixCreateFromMat(data_f, 1, 13, NAN, &dmat) != 0) return -7;

    bst_ulong out_len;
    const float* out;

    if (XGBoosterPredict(booster, dmat, 0, 0, 0, &out_len, &out) != 0) return -8;

    double prob = out[0];

    // ---------------- output features string ----------------

    std::vector<std::pair<std::string, float>> feats;

    for (int i = 0; i < 13; ++i)
        feats.push_back({feature_names[i], (float)data[i]});

    std::sort(feats.begin(), feats.end(),
        [](auto& a, auto& b){
            return std::abs(a.second) > std::abs(b.second);
        });

    std::stringstream ss;
    for (int i = 0; i < 5; i++) {
        ss << feats[i].first << ":" << feats[i].second;
        if (i != 4) ss << ", ";
    }

    std::string features_str = ss.str();
    strncpy(out_features, features_str.c_str(), out_size - 1);
    out_features[out_size - 1] = '\0';

    XGDMatrixFree(dmat);
    XGDMatrixFree(dmat_tmp);
    XGBoosterFree(booster);

    return prob <= LOW_THR ? 2 : (prob >= HIGH_THR ? 1 : 3);
}