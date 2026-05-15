g++ -shared trial5min.cpp -o trial.dll -I"xgboost/include" -I"xgboost/build32/dmlc-core/include" 
-L"xgboost/lib" "xgboost/lib/libxgboost.dll.a" "xgboost/build32/dmlc-core/libdmlc.dll.a"
-static-libgcc -static-libstdc++ -lws2_32 -lwinpthread -std=c++17
