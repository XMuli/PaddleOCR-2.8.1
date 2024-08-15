#pragma once
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif


struct OCRResult
{
    char** text_results;
    int* num_results;
    int num_images;
};

__declspec(dllexport) int add(int a, int b);
__declspec(dllexport) OCRResult* ImageProcess(const char* image_dir);
__declspec(dllexport) void FreeOCRResult(OCRResult* result);

#ifdef __cplusplus
}
#endif
