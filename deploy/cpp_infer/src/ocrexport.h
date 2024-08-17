#pragma once
#include <vector>
#include <string>


#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport) int add(int a, int b);
__declspec(dllexport) const char* ImageOcrProcess(const char* image_dir, const bool bSingular);
__declspec(dllexport) void FreeReturnPtr(const char* p); // �� exe �е����ͷţ�ȷ�� Debug ģʽ�ͷ� ImageOcrProcess ����ֵ�������

#ifdef __cplusplus
}
#endif
