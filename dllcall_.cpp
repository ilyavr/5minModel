#include <iostream>
#include <windows.h>

typedef int (*RunTrialClassFunc)(const char*, const char*, double, const char**);

int main() {
    HMODULE dll = LoadLibraryA("trial.dll");
    if (!dll) {
    DWORD err = GetLastError();
    std::cout << "Failed to load DLL. Error: " << err << std::endl;
    return 1;
    }

    RunTrialClassFunc run_trial_class =
        (RunTrialClassFunc)GetProcAddress(dll, "run_trial_class");

    if (!run_trial_class) {
        std::cout << "Function not found" << std::endl;
        return 1;
    }

    const char* top_features;
    int cls = run_trial_class("0418_model.json", "104.5,109,110.5,111.5", 30.7, &top_features);
    std::cout << "class = " << cls << std::endl;
    std::cout << "top features = " << top_features << std::endl;
    std::cout.flush(); 

    FreeLibrary(dll);
    return 0;
}