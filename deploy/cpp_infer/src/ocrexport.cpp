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

extern "C" {

    __declspec(dllexport) int add(int a, int b)
    {
        return a + b;
    }

    __declspec(dllexport) OCRResult* ImageProcess(const char* image_dir)
    {
        std::vector<cv::String> cv_all_img_names;
        cv::glob(image_dir, cv_all_img_names);
        std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

        if (cv_all_img_names.empty())
        {
            std::cerr << "[ERROR] No images found in the directory: " << image_dir << std::endl;
            return nullptr;
        }

        PPOCR ocr;
        std::vector<cv::Mat> img_list;
        std::vector<cv::String> img_names;

        for (int i = 0; i < cv_all_img_names.size(); ++i)
        {
            cv::Mat img = cv::imread(cv_all_img_names[i], cv::IMREAD_COLOR);
            if (!img.data)
            {
                std::cerr << "[ERROR] image read failed! image path: "
                    << cv_all_img_names[i] << std::endl;
                continue;
            }
            img_list.push_back(img);
            img_names.push_back(cv_all_img_names[i]);
        }

        std::vector<std::vector<OCRPredictResult>> ocr_results = ocr.ocr(img_list, true, true, false);

        //for (int i = 0; i < img_names.size(); ++i)
        //{
        //    std::cout << "Predict img: " << img_names[i] << std::endl;
        //    Utility::print_result(ocr_results[i]);
        //}

        OCRResult* result = new OCRResult;
        result->num_images = ocr_results.size();
        result->text_results = new char* [result->num_images];
        result->num_results = new int[result->num_images];

        for (size_t i = 0; i < ocr_results.size(); ++i)
        {
            // Allocate memory and store results (example: simple concatenation)
            std::string concatenated_text;
            for (const auto& item : ocr_results[i])
            {
                concatenated_text += item.text + " ";
            }

            result->text_results[i] = _strdup(concatenated_text.c_str());
            result->num_results[i] = ocr_results[i].size();
        }

        return result;
    }

    __declspec(dllexport) void FreeOCRResult(OCRResult* result)
    {
        if (result)
        {
            for (int i = 0; i < result->num_images; ++i)
            {
                free(result->text_results[i]);
            }
            delete[] result->text_results;
            delete[] result->num_results;
            delete result;
        }
    }
}
