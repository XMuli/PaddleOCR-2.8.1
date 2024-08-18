#include "ocrexport.h"
#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <iostream>
#include <windows.h>
#include <include/args.h>
#include <include/paddleocr.h>
#include <include/paddlestructure.h>

using namespace PaddleOCR;
#include "include/json.hpp"
using ordered_json = nlohmann::ordered_json;

extern "C" {

    // 将 ANSI 编码字符串转换为 UTF-8 编码字符串
    std::string ansiToUtf8(const std::string& ansiStr) {
        // 获取 ANSI 字符串的宽字符长度
        int wideCharLen = MultiByteToWideChar(CP_ACP, 0, ansiStr.c_str(), -1, NULL, 0);

        // 分配足够大的宽字符缓冲区
        std::wstring wideStr(wideCharLen, L'\0');

        // 将 ANSI 字符串转换为宽字符字符串
        MultiByteToWideChar(CP_ACP, 0, ansiStr.c_str(), -1, &wideStr[0], wideCharLen);

        // 获取 UTF-8 编码字符串的长度
        int utf8Len = WideCharToMultiByte(CP_UTF8, 0, wideStr.c_str(), -1, NULL, 0, NULL, NULL);

        // 分配足够大的 UTF-8 缓冲区
        std::string utf8Str(utf8Len, '\0');

        // 将宽字符字符串转换为 UTF-8 编码字符串
        WideCharToMultiByte(CP_UTF8, 0, wideStr.c_str(), -1, &utf8Str[0], utf8Len, NULL, NULL);

        return utf8Str;
    }

    void printMemory(const char* data, size_t size) {
        std::cout << "Memory content in hexadecimal:";
        for (size_t i = 0; i < size; ++i) {
            std::cout << std::hex << std::setw(2) << std::setfill('0') << (static_cast<unsigned int>(data[i]) & 0xFF) << ' ';
        }
        std::cout << std::dec << std::endl;  // Reset to decimal
    }

    void check_params()
    {
        if (FLAGS_det)
        {
            if (FLAGS_det_model_dir.empty() || FLAGS_image_dir.empty())
            {
                std::cout << "Usage[det]: ./ppocr "
                    "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                    << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
                exit(1);
            }
        }
        if (FLAGS_rec)
        {
            std::cout
                << "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320',"
                "if you are using recognition model with PP-OCRv2 or an older "
                "version, "
                "please set --rec_image_shape='3,32,320"
                << std::endl;
            if (FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty())
            {
                std::cout << "Usage[rec]: ./ppocr "
                    "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                    << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
                exit(1);
            }
        }
        if (FLAGS_cls && FLAGS_use_angle_cls)
        {
            if (FLAGS_cls_model_dir.empty() || FLAGS_image_dir.empty())
            {
                std::cout << "Usage[cls]: ./ppocr "
                    << "--cls_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                    << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
                exit(1);
            }
        }
        if (FLAGS_table)
        {
            if (FLAGS_table_model_dir.empty() || FLAGS_det_model_dir.empty() ||
                FLAGS_rec_model_dir.empty() || FLAGS_image_dir.empty())
            {
                std::cout << "Usage[table]: ./ppocr "
                    << "--det_model_dir=/PATH/TO/DET_INFERENCE_MODEL/ "
                    << "--rec_model_dir=/PATH/TO/REC_INFERENCE_MODEL/ "
                    << "--table_model_dir=/PATH/TO/TABLE_INFERENCE_MODEL/ "
                    << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
                exit(1);
            }
        }
        if (FLAGS_layout)
        {
            if (FLAGS_layout_model_dir.empty() || FLAGS_image_dir.empty())
            {
                std::cout << "Usage[layout]: ./ppocr "
                    << "--layout_model_dir=/PATH/TO/LAYOUT_INFERENCE_MODEL/ "
                    << "--image_dir=/PATH/TO/INPUT/IMAGE/" << std::endl;
                exit(1);
            }
        }
        if (FLAGS_precision != "fp32" && FLAGS_precision != "fp16" &&
            FLAGS_precision != "int8")
        {
            std::cout << "precison should be 'fp32'(default), 'fp16' or 'int8'. "
                << std::endl;
            exit(1);
        }
    }

    __declspec(dllexport) int add(int a, int b)
    {
        return a + b;
    }

    __declspec(dllexport) const char* ImageOcrProcess(const char* image_dir, const bool bSingular)
    {
        printMemory(image_dir, std::strlen(image_dir));
        PPOCR ocr;

        if (FLAGS_benchmark) 
        {
            ocr.reset_timer();
        }

        std::vector<cv::Mat> img_list;
        std::vector<cv::String> img_names;
        std::vector<cv::String> cv_all_img_names;  // 仅复数时候使用

        if (bSingular) {
            // 单张图片的情况，添加到 cv_all_img_names
            cv_all_img_names.push_back(image_dir);
        }
        else {
            // 多张图片的情况，使用 glob 获取所有图片路径
            cv::glob(image_dir, cv_all_img_names);
            std::cout << "Total images num: " << cv_all_img_names.size() << std::endl;
        }

            if (cv_all_img_names.empty())
            {
                std::cerr << "[ERROR] No images found in the directory: " << image_dir << std::endl;
                return "";
            }

            for (int i = 0; i < cv_all_img_names.size(); ++i)
            {
                cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
                if (!img.data)
                {
                    std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << std::endl;
                    continue;
                }
                img_list.push_back(img);
                img_names.push_back(cv_all_img_names[i]);
            }
  

        std::vector<std::vector<OCRPredictResult>> ocr_results = ocr.ocr(img_list, true, true, false);


        for (int i = 0; i < img_names.size(); ++i)  // 保存输出后的图片
        {
            //std::cout << "predict img: " << cv_all_img_names[i] << std::endl;
            //Utility::print_result(ocr_results[i]);
            if (FLAGS_visualize && FLAGS_det)
            {
                std::string file_name = Utility::basename(img_names[i]);
                cv::Mat srcimg = img_list[i];
                Utility::VisualizeBboxes(srcimg, ocr_results[i],
                    FLAGS_output + "/" + file_name);
            }
        }
        if (FLAGS_benchmark)
        {
            ocr.benchmark_log(cv_all_img_names.size());
        }

        ordered_json j;
        j["images"] = ordered_json::array();

        for (size_t i = 0; i < ocr_results.size(); ++i)
        {
            ordered_json image_json;
            image_json["index"] = i + 1;

            try { // 将 ANSI 编码字符串转换为 UTF-8 编码字符串
                std::string utf8Str = ansiToUtf8(img_names[i]); // ordered_json 需要传入 utf8 字符的
                image_json["path"] = utf8Str;
            }
            catch (const std::exception& e) { // 捕捉标准异常并输出错误信息
                std::cerr << "Exception: " << e.what() << std::endl;
            }
            catch (...)  // 捕捉所有其他类型的异常
            {
                std::cerr << "Unknown exception occurred." << std::endl;
            }

            image_json["text_results"] = ordered_json::array();

            for (size_t j = 0; j < ocr_results[i].size(); ++j)
            {
                ordered_json result_json;
                result_json["text"] = ocr_results[i][j].text;
                result_json["recognition_score"] = ocr_results[i][j].score;

                // Add detection box points
                ordered_json box_points = ordered_json::array();
                for (const auto& point : ocr_results[i][j].box)
                {
                    ordered_json point_json;
                    point_json["x"] = point[0];
                    point_json["y"] = point[1];
                    box_points.push_back(point_json);
                }
                result_json["detection_box"] = box_points;

                result_json["classification_label"] = ocr_results[i][j].cls_label;
                result_json["classification_score"] = ocr_results[i][j].cls_score;

                image_json["text_results"].push_back(result_json);
            }

            j["images"].push_back(image_json);
        }

        // Return the pointer to the allocated memory
        //std::cout << "--json-------->" << j.dump(4) << std::endl;
        std::string serialized_json = j.dump();

        char* result = new char[serialized_json.size() + 1];
        std::strcpy(result, serialized_json.c_str());

        //// 打印地址
        //std::cout << "Address of result: " << static_cast<void*>(result) << std::endl;
        //std::size_t size = std::strlen(result);
        //std::cout << "Size of result: " << size + 1 << " bytes" << std::endl;  // +1 包括 '\0'

        //// 打印内存内容
        //std::cout << "Memory content of result: ";
        //for (std::size_t i = 0; i < size + 1; ++i) {  // +1 包括 '\0'
        //    std::cout << "\\x" << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(static_cast<unsigned char>(result[i]));
        //}
        //std::cout << std::endl;

        return result;
    }



#include <iostream>
#include <cstring>  // For std::strlen

    void printMemory2(const char* data, size_t size) {
        std::cout << "Memory content in hexadecimal:" << std::endl;

        // Iterate over each byte in the data
        for (size_t i = 0; i < size; ++i) {
            // Print each byte in hexadecimal format with leading zeros and a space
            std::cout << std::hex << std::setw(2) << std::setfill('0')
                << static_cast<int>(static_cast<unsigned char>(data[i])) << " ";
        }

        std::cout << std::endl;  // Final newline
    }

    // 定义一个新接口来解析命令行参数
    __declspec(dllexport) const char* ImageOcrProcessWithArgs(const bool& bSingular, int argc, char** argv) {

        google::ParseCommandLineFlags(&argc, &argv, true);
        check_params();

        if (!Utility::PathExists(FLAGS_image_dir))
        {
            std::cerr << "[ERROR] image path not exist! image_dir: " << FLAGS_image_dir
                << std::endl;
            exit(1);
        }

        if (!Utility::PathExists(FLAGS_output))
        {
            Utility::CreateDir(FLAGS_output);
        }

        // 调用原有的 OCR 处理函数
        return ImageOcrProcess(FLAGS_image_dir.c_str(), bSingular);
    }

    __declspec(dllexport) void FreeReturnPtr(const char* p)
    {
        if (!p) delete[] p;
    }

}
