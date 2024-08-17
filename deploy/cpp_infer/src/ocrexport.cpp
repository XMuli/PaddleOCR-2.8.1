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


    __declspec(dllexport) int add(int a, int b)
    {
        return a + b;
    }

    __declspec(dllexport) const char* ImageOcrProcess(const char* image_dir, const bool bSingular)
    {
        PPOCR ocr;
        std::vector<cv::Mat> img_list;
        std::vector<cv::String> img_names;

        if (bSingular)
        {
            cv::Mat img = cv::imread(image_dir, cv::IMREAD_COLOR);
            if (!img.data) {
                std::cerr << "[ERROR] Image read failed! Image path: " << image_dir << std::endl;
                return "";
            }
            img_list.push_back(img);
            img_names.push_back(image_dir);
        }
        else
        {
            std::vector<cv::String> cv_all_img_names;
            cv::glob(image_dir, cv_all_img_names);
            std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

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
        }

        std::vector<std::vector<OCRPredictResult>> ocr_results = ocr.ocr(img_list, true, true, false);

        ordered_json j;
        j["images"] = ordered_json::array();

        for (size_t i = 0; i < ocr_results.size(); ++i)
        {
            ordered_json image_json;
            image_json["index"] = i + 1;
            image_json["path"] = img_names[i];
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

    __declspec(dllexport) void FreeReturnPtr(const char* p)
    {
        if (!p) delete[] p;
    }

}
