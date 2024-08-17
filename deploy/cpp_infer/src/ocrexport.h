#pragma once
#include <vector>
#include <string>


#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)  // 强制1字节对齐
struct OCRResult
{
    std::vector<std::vector<std::vector<std::vector<int>>>> boxes; // Detection boxes   共2张图/第1张有2条识别信息/每条识别信息有指定的的原始数据
    std::vector<std::vector<std::string>> text_results; // Recognized text
    std::vector<std::vector<float>> rec_scores; // Recognition scores
    std::vector<std::vector<int>> cls_labels; // Classification labels
    std::vector<std::vector<float>> cls_scores; // Classification scores
    std::vector<std::string> path;
    int num_images;
};

#pragma pack(pop)  // 恢复默认对齐


__declspec(dllexport) int add(int a, int b);
__declspec(dllexport) OCRResult* ImageProcess(const char* image_dir);
__declspec(dllexport) const char* ImageProcess2(const char* image_dir);
__declspec(dllexport) void FreeOCRResult(OCRResult* result);


#ifdef __cplusplus
}
#endif
