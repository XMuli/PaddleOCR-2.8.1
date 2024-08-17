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
#include "json.hpp"
using ordered_json = nlohmann::ordered_json;

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
                std::cerr << "[ERROR] image read failed! image path: " << cv_all_img_names[i] << std::endl;
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
        
        result->boxes.resize(result->num_images);
        result->path.resize(result->num_images);
        result->text_results.resize(result->num_images);
        result->rec_scores.resize(result->num_images);
        result->cls_labels.resize(result->num_images);
        result->cls_scores.resize(result->num_images);
        result->path = img_names;



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

        std::cout << "-------------DLL---------->" << std::endl
        << "result: " << result  << "  result->num_images: " << result->num_images << "  img_names.size():" << img_names.size() << std::endl;
        for (int i = 0; i < result->path.size(); ++i) std::cout << "result->path: " << result->path[i] << std::endl;

        // �� DLL �� EXE �ڲ��ֱ�ִ��
        std::cout << std::endl << "Size of OCRResult: " << sizeof(OCRResult) << std::endl;
        std::cout << "Offset of num_images: " << offsetof(OCRResult, num_images) << std::endl;
        
        unsigned char* raw_data = reinterpret_cast<unsigned char*>(result);    // Ҳ�������ֽڴ�ӡ OCRResult ���ڴ�����
        for (size_t i = 0; i < sizeof(OCRResult); ++i) std::cout << std::hex << (int)raw_data[i] << " ";
        std::cout << std::endl;

#if 0
        // ����Ĵ�ӡ��Ϣ
        for (int i = 0; i < result->num_images; ++i) {
            std::cout << "Image " << i << "  result->path: " << result->path[i] << " OCR Results:" << std::endl;

            // ��ӡʶ����ı�����Ͷ�Ӧ��ʶ�����
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

#endif

        return result;
    }

    __declspec(dllexport) const char* ImageProcess2(const char* image_dir)
    {
        
        std::vector<cv::String> cv_all_img_names;
        cv::glob(image_dir, cv_all_img_names);
        std::cout << "total images num: " << cv_all_img_names.size() << std::endl;

        if (cv_all_img_names.empty())
        {
            std::cerr << "[ERROR] No images found in the directory: " << image_dir << std::endl;
            return "";
        }

        PPOCR ocr;
        std::vector<cv::Mat> img_list;
        std::vector<cv::String> img_names;

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
        std::cout << "--json-------->" << j.dump(4) << std::endl;
        // Serialize the JSON object to a string
        std::string serialized_json = j.dump();

        // Allocate memory for the C-style string and copy the content
        char* result = new char[serialized_json.size() + 1];
        std::strcpy(result, serialized_json.c_str());



        return result;
    }

    __declspec(dllexport) void FreeOCRResult(OCRResult* result)
    {
        if (result)
        {
            // ������� vectors ���ͷ��ڴ�
            for (int i = 0; i < result->num_images; ++i)
            {
                // Clear nested vectors for each image
                for (auto& boxVec : result->boxes[i])
                    boxVec.clear(); // Clear inner vectors
                result->boxes[i].clear();  // Clear outer vector

                result->text_results[i].clear();  // ���ÿ��ͼƬ���ı�ʶ����
                result->rec_scores[i].clear();  // ���ÿ��ͼƬ��ʶ�����
                result->cls_labels[i].clear();  // ���ÿ��ͼƬ�ķ����ǩ
                result->cls_scores[i].clear();  // ���ÿ��ͼƬ�ķ������
            }

            // Clear overall vectors
            result->boxes.clear();
            result->text_results.clear();
            result->rec_scores.clear();
            result->cls_labels.clear();
            result->cls_scores.clear();
            result->path.clear(); // Clear the path vector

            // Delete the OCRResult object itself
            delete result;
        }
    }


}
