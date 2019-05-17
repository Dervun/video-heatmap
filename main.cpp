/*!
 * Программа для построения тепловых карт по видео.
 */
#include <iostream>
#include <vector>
#include <math.h>
#include <exception>
#include <fstream>
#include <deque>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaarithm.hpp>

/// Флаги компиляции, для более или менее подробного вывода
#define SHOW_ORIGIN
#define SHOW_THRESHOLD
#define SHOW_CONTOURS
#define PRINT_PROGRESS_OF_TRACK

#ifdef PRINT_PROGRESS_OF_TRACK
#define PRINT_STEP 5
#endif

using namespace std;


/*!
 * \brief The cantOpenVideoForReading class
 * Специальное исключение, которое бросается в том случае, если не удаётся открыть файл с видео.
 */
class cantOpenVideoForReading : public exception
{
    const char* what() const noexcept
    { return "Can't open video for reading"; }
} openVideoException;


/*!
 * \brief rotatePoints
 * \param a Первая точка отрезка
 * \param b Вторая точка отрезка
 * \param c Точка, для которой определяется положение
 * \return Значение больше 0 если точка c лежит слева от прямой (a, b),\
 *     значение меньше 0, если точка c лежит справа от прямой (a, b),\
 *     значение равно нулю, если точка c лежит на прямой (a, b).
 */
inline int rotatePoints(const pair<int, int> a, const pair<int, int> b, const pair<int, int> c)
{
    return (b.first - a.first) * (c.second - b.second) - (b.second - a.second) * (c.first - b.first);
}


/*!
 * \brief checkPointInConvexHull
 * \param point Точка, которая локализуется
 * \param hull Выпуклая оболочка, нахождение в которой проверяется
 * \return true если точка point лежит в выпуклой оболочке hull
 * \warning Проверка производится с учётом, что точки в hull расположены против часовой стрелки с учётом традиционных математических осей координат,\
 *     то есть если координата y отсчитывается от верхнего края, то точки должны быть расположены по часовой стрелке.
 * С помощью бинарного поиска найдём сектор (hull[i], hull[0], hull[i+1]), в котором находится точка,
 * и проверим, лажит ли она внутри этого сектора.
 * Более подробно: https://habr.com/ru/post/144571/
 */
bool checkPointInConvexHull(const pair<int, int> point, const vector<cv::Point>& hull)
{
    if (hull.size() < 3)
        return false;

    const pair<int, int> begin = make_pair(hull[0].x, hull[0].y);
//    cout << "hull[0]: " << hull[0] << ", hull[1]: " << hull[1] << ", hull[" << hull.size() - 1 << "]: " << hull[hull.size()-1] << endl;

    // Если точка не лежит в самом большом секторе, то точно не лежит в оболочке
    if (rotatePoints(begin, make_pair(hull[1].x, hull[1].y), point) < 0
            || rotatePoints(begin, make_pair(hull[hull.size()-1].x, hull[hull.size()-1].y), point) > 0)
        return false;

    int left = 1;
    int right = hull.size() - 1;
    while (left <= right)
    {
//        cout << "left = " << hull[left] << ", right = " << hull[right] << endl;
        int middle = left + (right - left) / 2;
        if (rotatePoints(begin, make_pair(hull[middle].x, hull[middle].y), point) > 0)
            left = middle + 1;
        else
            right = middle - 1;
    }

    // Теперь в left лежит индекс первого луча (hull[0], hull[left]), от которого точка point лежит не слева (справа или на нём).
    // Получается, что искомое i для сектора (hull[i], hull[0], hull[i+1]), в котором находится точка, это left - 1.
    // Проверим, что point находится внутри оболочки, то есть справа от ребра.
    const int i = left - 1;
//    cout << "found hull[i] = " << hull[i] << endl;
    return rotatePoints(make_pair(hull[i].x, hull[i].y), make_pair(hull[i+1].x, hull[i+1].y), point) >= 0;
}


/*!
 * \brief testCheckPointInConvexHull
 * Простой тест, проверяющий корректную работу функции локализации точки в выпуклом многоугольнике checkPointInConvexHull.
 */
bool testCheckPointInConvexHull(const bool verbose = false)
{
    vector<cv::Point> hull = {cv::Point(2, 3), cv::Point(5, 2), cv::Point(8, 2), cv::Point(10, 5), cv::Point(8, 8), cv::Point(3, 8)};
    vector<pair<int, int>> valid = {make_pair(5, 3), make_pair(6, 2), make_pair(8, 4), make_pair(8, 6), make_pair(5, 8), make_pair(3, 5)};
    vector<pair<int, int>> invalid = {make_pair(3, 2), make_pair(4, 2), make_pair(10, 3), make_pair(10, 7), make_pair(5, 9), make_pair(1, 1)};

    if (verbose)
    {
        cout << "valid: ";
        cout << std::noboolalpha;
    }
    bool allAreValid = true;
    for (const pair<int, int>& point : valid)
    {
        bool currentResult = checkPointInConvexHull(point, hull);
        allAreValid &= currentResult;
        if (verbose)
            cout << currentResult << ' ';
    }
    if (verbose)
    {
        cout << endl << "all are valid: " << std::boolalpha << allAreValid << endl << endl;

        cout << "invalid: ";
        cout << std::noboolalpha;
    }
    bool allAreInvalid = true;
    for (const pair<int, int>& point : invalid)
    {
        bool currentResult = checkPointInConvexHull(point, hull);
        allAreInvalid &= !currentResult;
        if (verbose)
            cout << currentResult << ' ';
    }
    if (verbose)
        cout << endl << "all are invalid: " << std::boolalpha << allAreValid << endl;
    return allAreValid && allAreInvalid;
}


/*!
 * \brief The Blob struct
 * Вспомогательная структура для хранения информации об области движения.
 */
struct Blob
{
    Blob(vector<cv::Point> contour)
        : contour(contour)
    {
        boundingRect = cv::boundingRect(contour);
        x = boundingRect.x;
        y = boundingRect.y;
        width = boundingRect.width;
        height = boundingRect.height;
        x_end = x + width - 1;
        y_end = y + height - 1;
        area = boundingRect.area();
        diagonalSize = sqrt(pow(width, 2) + pow(height, 2));
        aspectRatio = static_cast<double>(width) / height;
    }

    cv::Rect boundingRect;  /// Обрамляющий прямоугольник
    vector<cv::Point> contour;  /// Контур области движения (многоугольник при последовательном соединении)
    int x = 0;   /// Координаты верхнего левого края
    int y = 0;
    int width = 0;  /// Ширина и высота обрамляющего прямоугольника
    int height = 0;
    int x_end = 0;  /// Координаты правого нижнего края
    int y_end = 0;
    int area = 0;  /// Площать обрамляющего многоугольника
    double diagonalSize = 0;  /// Длина диагонали обрамляющего многоугольника
    double aspectRatio = 0;  /// Отношение ширины к высоте
};

#ifdef SHOW_CONTOURS
void drawAndShowContours(const cv::Size &imgSize, const vector<vector<cv::Point>> &contours)
{
    cv::Mat image = cv::Mat::zeros(imgSize, CV_8UC1);
    cv::drawContours(image, contours, -1, cv::Scalar(255), -1);
    cv::imshow("Contours", image);
}
#endif

/*!
 * \brief detector
 * \param img1 Первое изображение
 * \param img2 Второе изображение
 * \param useConvexHull Флаг, указывающий что необходимо найти выпуклую оболочку контура
 * \return Список областей движения
 * Функция используется для поиска движений (изменений) между двумя кадрами видео.
 */
vector<Blob> detector(const cv::cuda::GpuMat &img1, const cv::cuda::GpuMat &img2, bool useConvexHull, const int binThreshold = 10)
{
    vector<Blob> result;

    cv::cuda::GpuMat imgDiff;
    cv::cuda::absdiff(img1, img2, imgDiff);
    cv::cuda::GpuMat imgThresh;
    cv::cuda::threshold(imgDiff, imgThresh, binThreshold, 255.0, CV_THRESH_BINARY);

    const cv::Mat structuringElementNxN = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    const cv::Ptr<cv::cuda::Filter> dilateFilter = cv::cuda::createMorphologyFilter(cv::MORPH_DILATE, imgThresh.type(), structuringElementNxN);
    const cv::Ptr<cv::cuda::Filter> erodeFilter = cv::cuda::createMorphologyFilter(cv::MORPH_ERODE, imgThresh.type(), structuringElementNxN);
    // Apply dilate, erode and erode, dilate
    // try in-place computation
    dilateFilter->apply(imgThresh, imgDiff);
    erodeFilter->apply(imgDiff, imgThresh);
    erodeFilter->apply(imgThresh, imgDiff);
    dilateFilter->apply(imgDiff, imgThresh);
    #ifdef SHOW_THRESHOLD
    cv::imshow("Threshold", imgThresh);
    #endif
    cv::Mat imgTheshCopy;
    imgThresh.download(imgTheshCopy);

    vector<vector<cv::Point>> contours;
    cv::findContours(imgTheshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    vector<vector<cv::Point>> hulls;
    if (useConvexHull)
    {
        hulls.resize(contours.size());
        for (size_t i = 0; i < contours.size(); i++)
            cv::convexHull(contours[i], hulls[i]);
    }
//    #ifdef SHOW_CONTOURS
//    drawAndShowContours(imgThresh.size(), hulls);
//    #endif

    const int areaThreshold = 1000;
    const int minAspectRatio = 0.1;
    const int maxAspectRatio = 2.0;
    for (size_t i = 0; i < contours.size(); i++)
    {
        Blob possibleBlob(contours[i]);
        if (useConvexHull)
            possibleBlob = Blob(hulls[i]);

        if (possibleBlob.area > areaThreshold
                && possibleBlob.aspectRatio > minAspectRatio
                && possibleBlob.aspectRatio < maxAspectRatio)
//                && (possibleBlob.width > 30 || possibleBlob.height > 30)
//                && cv::contourArea(possibleBlob.contour) / possibleBlob.area > 0.50)
            result.push_back(possibleBlob);
    }

    #ifdef SHOW_CONTOURS
    vector<vector<cv::Point>> blobCountors(result.size());
    for (size_t i = 0; i < result.size(); i++)
        blobCountors[i] = result[i].contour;
    drawAndShowContours(imgThresh.size(), blobCountors);
    #endif

    return result;
}


/*!
 * \brief getHeatmap
 * \param videoName Имя файла с видеозаписью, по которому необходимо построить тепловую карту
 * \param useConvexHull Флаг, назначающий использование инкрементирования только внутри выпуклой оболочки, а не внутри обрамляющего многоугольника
 * \param frameStep Смотрится движение между двумя кадрами, отстоящими между собой на frameStep - 1 позиций
 * \param saveStep Каждые сколько секунд необходимо сохранять текущее состояние счётчиков в файл
 * \param indent На сколько секунд от начала файла нужно его промотать, прежде чем начать построение карты
 * \param binThreshold Планка, значение, после которого изменение в цвете пикселя начинает учитываться
 * \return Возвращает построенную тепловую карту, то есть счётчики движений по пикселям
 */
vector<vector<unsigned int>> getHeatmap(const cv::String &videoName, const bool useConvexHull = false,
                                        const int frameStep = 12, const int saveStep = INT_MAX,
                                        const int indent = 0, const int binThreshold = 10)
{
    cv::VideoCapture cap(videoName);  // open the video file for reading
    if (!cap.isOpened())  // if not success, exit function
    {
        cout << "Cannot open the video file" << endl;
        throw openVideoException;
    }

    ofstream result_file(videoName + "_result.txt", ios_base::out);

    // read some data from input file
    int fourcc = cap.get(CV_CAP_PROP_FOURCC);
    fourcc = 0x21;  // иначе будет ошибка на Linux
    printf("CAP_PROP_FOURCC: %c%c%c%c\n", fourcc & 255, (fourcc >> 8) & 255, (fourcc >> 16) & 255, (fourcc >> 24) & 255);

    double fps = cap.get(CV_CAP_PROP_FPS);  // get the frames per seconds of the video
    cout << "Frame per seconds : " << fps << endl;

    cv::Size frameSize = cv::Size(cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
    cout << "Frame size: " << frameSize << endl;
    const int height = frameSize.height;
    const int width = frameSize.width;

    double frameCount = cap.get(CV_CAP_PROP_FRAME_COUNT);
    cout << "frameCount: " << frameCount << endl;

    vector<vector<unsigned int>> counter(height, vector<unsigned int>(width, 0));
    result_file << width << ' ' << height << endl;

    #ifdef SHOW_ORIGIN
    cv::namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);
    #endif
    #ifdef SHOW_THRESHOLD
    cv::namedWindow("Threshold", CV_WINDOW_OPENGL);
    #endif
    #ifdef SHOW_CONTOURS
    cv::namedWindow("Contours", CV_WINDOW_AUTOSIZE);  // CV_WINDOW_OPENGL
    #endif

    cv::cuda::GpuMat colorFrame2;
    cv::cuda::GpuMat grayFrame2;
    cv::Mat imgFrame2;
//    cv::Mat imgFrame3;

    bool success = true;
    for (double i = 0; i < indent * fps && success; i++)
        success = cap.read(imgFrame2);
    if (!success)
    {
        cout << "End of file :)" << endl;
        result_file.close();
        return counter;
    }

    cap.read(imgFrame2);
//    cv::resize(imgFrame3, imgFrame2, cv::Size(), 0.25, 0.25);
    colorFrame2.upload(imgFrame2);

    // cv::Size(x, y) -- x and y can differ but they both must be positive and odd
    const int gaussianKernel = 5;
    const cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(grayFrame2.type(), grayFrame2.type(), cv::Size(gaussianKernel, gaussianKernel), 0);
    cv::cuda::cvtColor(colorFrame2, grayFrame2, CV_BGR2GRAY);
    gaussianFilter->apply(grayFrame2, colorFrame2);

    cv::Mat imgFrame1;
    cv::cuda::GpuMat colorFrame1;
    cv::cuda::GpuMat grayFrame1;

    for (int i = 0; ; i++)
    {
        cv::swap(imgFrame1, imgFrame2);
        cv::cuda::swap(colorFrame1, colorFrame2);
        cv::cuda::swap(grayFrame1, grayFrame2);
        bool bSuccess = true;
        for (int j = 0; j < frameStep && bSuccess; j++)  // if not success, break loop
            bSuccess = cap.read(imgFrame2);  // read a new frame from video
        if (!bSuccess)
        {
            cout << "End of file :)" << endl;
            break;
        }
//        cv::resize(imgFrame3, imgFrame2, cv::Size(), 0.25, 0.25);
        colorFrame2.upload(imgFrame2);

        cv::cuda::cvtColor(colorFrame2, grayFrame2, CV_BGR2GRAY);

        gaussianFilter->apply(grayFrame2, colorFrame2);

        vector<Blob> currentBlobs = detector(colorFrame1, colorFrame2, useConvexHull, binThreshold);
        // Запишем извенения в счётчики
        if (useConvexHull)
        {
            vector<cv::Point> currentHull;
            deque<pair<int, int>> q;
            set<pair<int, int>> used;
            for (size_t i = 0; i < currentBlobs.size(); i++)
            {
                currentHull = currentBlobs[i].contour;
//                cout << "\nConvex hull:\n";
//                for (cv::Point p : currentHull)
//                    cout << "(" << p.x << ", " << p.y << "); ";
                // Обход в ширину с проверкой, что точка в выпуклой оболочке
                q.clear();
                used.clear();

                used.insert(make_pair(currentHull[0].x, currentHull[0].y));
                q.push_back(make_pair(currentHull[0].x, currentHull[0].y));

                while (!q.empty())
                {
                    const pair<int, int> currentPoint = q.front();
                    q.pop_front();
                    const int x = currentPoint.first;
                    const int y = currentPoint.second;
                    counter[y][x]++;
                    const vector<pair<int, int>> candidates = {make_pair(x + 1, y), make_pair(x - 1, y),
                                                               make_pair(x, y + 1), make_pair(x, y - 1)};
                    for (const pair<int, int>& candidate : candidates)
                    {
                        if (used.count(candidate) == 0 && candidate.first >= 0 && candidate.first < width
                                && candidate.second >= 0 && candidate.second <= height
                                && checkPointInConvexHull(candidate, currentHull))
                        {
                            used.insert(candidate);
                            q.push_back(candidate);
                        }
                    }
                }
            }
        }
        else
        {
            for (size_t i = 0; i < currentBlobs.size(); i++)
            {
                int x_begin = currentBlobs[i].x;
                int y_begin = currentBlobs[i].y;
                int x_end = currentBlobs[i].x_end;
                int y_end = currentBlobs[i].y_end;
                for (int y = y_begin; y <= y_end; y++)
                    for (int x = x_begin; x <= x_end; x++)
                        counter[y][x]++;
            }
        }

        if (static_cast<double>(i) * frameStep > static_cast<double>(saveStep) * fps)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                    result_file << counter[y][x] << ' ';
                result_file << "\r\n";
            }
            result_file << "#" << saveStep << "\r\n";
            i = 0;
        }

        #if defined(SHOW_ORIGIN) || defined(SHOW_THRESHOLD) || defined(SHOW_CONTOURS)
        cv::imshow("MyVideo", imgFrame2);  // show the frame in "MyVideo" window
        if (cv::waitKey(30) == 27)  // wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
        {
            cout << "esc key is pressed by user" << endl;
            break;
        }
        #endif
        #ifdef PRINT_PROGRESS_OF_TRACK
        if (!(i % PRINT_STEP))
            cout << "Processed " << cap.get(CV_CAP_PROP_POS_FRAMES) / frameCount * 100 << " %" << endl;
        #endif
    }    

    result_file.close();
    return counter;
}


int main(int argc, char *argv[])
{
//    cv::cuda::printCudaDeviceInfo(cv::cuda::getDevice());
//    cout << cv::getBuildInformation() << endl;

//    assert(testCheckPointInConvexHull(true));

    const cv::String input = "test_heatmaps.mp4";
    vector<vector<unsigned int>> result;
    try
    {
        result = getHeatmap(input, false);
    }
    catch (exception& e)
    {
        cout << "In track(\"" << input << "\") throwed exception: " << e.what() << endl;
        return 1;
    }

    // write results to file
    ofstream result_file(input + "_result.txt", ios_base::app);
    for (int y = 0; y < result.size(); y++)
    {
        for (int x = 0; x < result[0].size(); x++)
            result_file << result[y][x] << ' ';
        result_file << "\r\n";
    }
    result_file.close();
    cout << "results have been saved to " << input << "_result.txt" << endl;

    return 0;
}
