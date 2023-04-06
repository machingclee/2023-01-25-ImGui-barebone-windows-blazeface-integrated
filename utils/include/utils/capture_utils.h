#pragma once
#include "torch/torch.h"
#include "pch.h"
#include <Windows.h>
#include <iostream>
#include <string>

namespace CaptureUtils {
class hwnd2Mat {
public:
    hwnd2Mat(HWND hwindow, float scale = 1);
    virtual ~hwnd2Mat();
    virtual void read();
    cv::Mat image;

private:
    HWND hwnd;
    HDC hwindowDC, hwindowCompatibleDC;
    int height, width, srcheight, srcwidth;
    HBITMAP hbwindow;
    BITMAPINFOHEADER bi;
};

int start_webcam_capture(int camera_index);
int start_screen_capture(std::string filename);
} // namespace CaptureUtils
