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

        for (int i = 0; i < img_names.size(); ++i)
        {
            std::cout << "Predict img: " << img_names[i] << std::endl;
            Utility::print_result(ocr_results[i]);
        }


        OCRResult* result = new OCRResult;
        result->num_images = ocr_results.size();
        std::cout << "-------------DLL---------->" << result->num_images << "  " << img_names.size() << std::endl;
        result->boxes.resize(result->num_images);
        result->path.resize(result->num_images);
        result->text_results.resize(result->num_images);
        result->rec_scores.resize(result->num_images);
        result->cls_labels.resize(result->num_images);
        result->cls_scores.resize(result->num_images);

        result->path = img_names; // 文件名

        for (int i = 0; i < result->path.size(); ++i)
        {
            std::cout << "result->path: " << result->path[i] << std::endl;
        }

        for (size_t i = 0; i < ocr_results.size(); ++i)
        {
            result->boxes[i].resize(ocr_results[i].size());
            result->text_results[i].resize(ocr_results[i].size());
            result->rec_scores[i].resize(ocr_results[i].size());
            result->cls_labels[i].resize(ocr_results[i].size());
            result->cls_scores[i].resize(ocr_results[i].size());


            for (size_t j = 0; j < ocr_results[i].size(); ++j)
            {
                result->boxes[i][j].resize(ocr_results[i][j].box.size());
                result->boxes[i][j] = ocr_results[i][j].box;
                

                result->text_results[i][j] = ocr_results[i][j].text;
                result->rec_scores[i][j] = ocr_results[i][j].score;
                result->cls_labels[i][j] = ocr_results[i][j].cls_label;
                result->cls_scores[i][j] = ocr_results[i][j].cls_score;
            }
        }


        // 在 DLL 和 EXE 内部分别执行
        std::cout << std::endl << "Size of OCRResult: " << sizeof(OCRResult) << std::endl;
        std::cout << "Offset of num_images: " << offsetof(OCRResult, num_images) << std::endl;
        // 也可以逐字节打印 OCRResult 的内存内容
        unsigned char* raw_data = reinterpret_cast<unsigned char*>(result);
        for (size_t i = 0; i < sizeof(OCRResult); ++i) {
            std::cout << std::hex << (int)raw_data[i] << " ";
        }
        std::cout << std::endl;


        // 打印 OCRResult 中的所有内容
        std::cout << result  << "  OCRResult contains " << result->num_images << " images:\n";



        for (int i = 0; i < result->num_images; ++i) {
            std::cout << "Image " << i << "  result->path: " << result->path[i] << " OCR Results:" << std::endl;

            // 打印识别的文本结果和对应的识别分数
            for (size_t j = 0; j < result->text_results[i].size(); ++j) {
                std::cout << "\tText: " << result->text_results[i][j] << std::endl;
                std::cout << "\tRecognition Score: " << result->rec_scores[i][j] << std::endl;
            }
        }

        for (int i = 0; i < result->num_images; ++i)
        {
            std::cout << "Image " << i << " OCR Results:\n";
            for (size_t j = 0; j < result->boxes[i].size(); ++j)
            {
                std::cout << "\tdet boxes: [";
                for (const auto& point : result->boxes[i][j])
                {
                    std::cout << "[" << point[0] << "," << point[1] << "],";
                }
                std::cout << "]\n";
                std::cout << "\trec text: " << result->text_results[i][j] << "\n";
                std::cout << "\trec score: " << result->rec_scores[i][j] << "\n";
                if (result->cls_labels[i][j])
                {
                    std::cout << "\tcls label: " << result->cls_labels[i][j] << "\n";
                    std::cout << "\tcls score: " << result->cls_scores[i][j] << "\n";
                }
            }
        }

        return result;
    }


    __declspec(dllexport) void FreeOCRResult(OCRResult* result)
    {
        if (result)
        {
            // 清空所有 vectors 以释放内存
            for (int i = 0; i < result->num_images; ++i)
            {
                result->boxes[i].clear();  // 清空每个图片的 box 数据
                result->text_results[i].clear();  // 清空每个图片的文本识别结果
                result->rec_scores[i].clear();  // 清空每个图片的识别分数
                result->cls_labels[i].clear();  // 清空每个图片的分类标签
                result->cls_scores[i].clear();  // 清空每个图片的分类分数
            }

            // 清空整体 vectors
            result->boxes.clear();
            result->text_results.clear();
            result->rec_scores.clear();
            result->cls_labels.clear();
            result->cls_scores.clear();

            // 删除 OCRResult 对象本身
            delete result;
        }
    }


}
