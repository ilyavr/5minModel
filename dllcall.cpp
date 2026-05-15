#include <iostream>
#include <windows.h>


typedef int (*RunTrialClassFunc)(
    const char*,   // model_path
    const char*,   // power
    double,        // temperature
    char*,         // буфер
    int            // размер буфера
);

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
        FreeLibrary(dll);
        return 1;
    }

    char str[] = "73.5,76,78,82,84,87.5";

    // буфер под строку
    char buffer[512];

    int cls = run_trial_class(
        "SZ65_0551_model.json",
        str,
        21.0,
        buffer,
        sizeof(buffer)
    );

    std::cout << "class = " << cls << std::endl;
    std::cout << "top features = " << buffer << std::endl;

    std::cout.flush();

    FreeLibrary(dll);
    return 0;
}