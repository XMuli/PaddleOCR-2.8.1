#pragma once
#include <vector>
#include <string>


#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(push, 1)  // ǿ��1�ֽڶ���
struct OCRResult
{
    std::vector<std::vector<std::vector<std::vector<int>>>> boxes; // Detection boxes   ��2��ͼ/��1����2��ʶ����Ϣ/ÿ��ʶ����Ϣ��ָ���ĵ�ԭʼ����
    std::vector<std::vector<std::string>> text_results; // Recognized text
    std::vector<std::vector<float>> rec_scores; // Recognition scores
    std::vector<std::vector<int>> cls_labels; // Classification labels
    std::vector<std::vector<float>> cls_scores; // Classification scores
    std::vector<std::string> path;
    int num_images;
};

#pragma pack(pop)  // �ָ�Ĭ�϶���


__declspec(dllexport) int add(int a, int b);
__declspec(dllexport) OCRResult* ImageProcess(const char* image_dir);
__declspec(dllexport) const char* ImageProcess2(const char* image_dir);
__declspec(dllexport) void FreeOCRResult(OCRResult* result);


#ifdef __cplusplus
}
#endif
