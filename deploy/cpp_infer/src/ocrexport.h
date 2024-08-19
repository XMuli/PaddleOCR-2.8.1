#pragma once
#include <vector>
#include <string>


#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport) int add(int a, int b);

__declspec(dllexport) const char* ImageOcrProcess(const char* image_dir);
__declspec(dllexport) const char* ImageOcrProcessWithArgs(int argc, char** argv);
__declspec(dllexport) void FreeReturnPtr(const char* p); // 供 exe 中调用释放，确保 Debug 模式释放 ImageOcrProcess 返回值不会崩溃

#ifdef __cplusplus
}
#endif
