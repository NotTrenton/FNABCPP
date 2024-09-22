// main.cpp

#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <string>
#include <vector>
#include <mutex>

// Serial Communication
#include <boost/asio.hpp>
#include <boost/asio/serial_port.hpp>

// OpenCV
#include <opencv2/opencv.hpp>

// libtorch
#include <torch/script.h> // One-stop header.

// GUI and Hotkey (using SDL2 for GUI)
#include <SDL2/SDL.h>

// Platform-specific includes for hotkey listening
#ifdef _WIN32
#include <windows.h>
#else
// Include headers for Linux or macOS if needed
#endif

using namespace std;
using namespace cv;

class Kmbox {
public:
    Kmbox() : io(), port(io) {
        detect_kmbox_port();
        if (!kmbox_port.empty()) {
            try {
                port.open(kmbox_port);
                port.set_option(boost::asio::serial_port_base::baud_rate(115200));
                cout << "[INFO] KMBOX connected on " << kmbox_port << endl;
            } catch (std::exception& e) {
                cerr << "[ERROR] Could not connect to KMBOX on " << kmbox_port << ". " << e.what() << endl;
                close_connection();
            }
        } else {
            cerr << "[ERROR] Could not find KMBOX device." << endl;
        }
    }

    ~Kmbox() {
        close_connection();
    }

    void close_connection() {
        if (port.is_open()) {
            port.close();
        }
    }

    void move(int x, int y, int steps = 0) {
        send_kmbox_command("km.move(" + to_string(x) + "," + to_string(y) + "," + to_string(steps) + ")");
    }

    void send_click() {
        send_kmbox_command("km.click(0)");
    }

private:
    boost::asio::io_service io;
    boost::asio::serial_port port;
    string kmbox_port;

    void detect_kmbox_port() {
        // Platform-specific port detection
#ifdef _WIN32
        // Windows implementation
        for (int i = 1; i <= 256; ++i) {
            string port_name = "\\\\.\\COM" + to_string(i);
            try {
                boost::asio::serial_port test_port(io, port_name);
                test_port.set_option(boost::asio::serial_port_base::baud_rate(115200));
                send_kmbox_command("km.version()", test_port);
                // Simple way to test if KMBOX responds
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (test_port.is_open()) {
                    kmbox_port = port_name;
                    test_port.close();
                    break;
                }
            } catch (...) {
                // Ignore exceptions and continue
            }
        }
#else
        // Implement for Linux or macOS
#endif
    }

    void send_kmbox_command(const string& command, boost::asio::serial_port& serial_port) {
        if (serial_port.is_open()) {
            string cmd = command + "\r\n";
            boost::asio::write(serial_port, boost::asio::buffer(cmd.c_str(), cmd.size()));
        } else {
            cerr << "[ERROR] KMBOX not connected." << endl;
        }
    }

    void send_kmbox_command(const string& command) {
        send_kmbox_command(command, port);
    }
};

void detector_thread(std::atomic<bool>& run_event, std::atomic<bool>& detector_running, Kmbox& kmbox, Rect detection_area) {
    // Load the YOLOv5 model
    torch::jit::script::Module model;
    try {
        model = torch::jit::load("best.pt");
        cout << "[INFO] Model loaded successfully." << endl;
    } catch (const c10::Error& e) {
        cerr << "[ERROR] Could not load model. " << e.what() << endl;
        return;
    }

    // Set device to CPU
    torch::Device device(torch::kCPU);
    model.to(device);

    // Initialize video capture from capture card
    int device_index = 0; // Change this to the index of your capture card
    VideoCapture cap(device_index);
    if (!cap.isOpened()) {
        cerr << "[ERROR] Could not open video capture device." << endl;
        return;
    }

    while (true) {
        if (!run_event.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        Mat frame;
        cap >> frame;
        if (frame.empty()) {
            cerr << "[ERROR] Failed to read frame from capture card." << endl;
            break;
        }

        // Crop the detection area from the frame
        Mat cropped_frame = frame(detection_area);

        // Convert image to tensor
        cv::cvtColor(cropped_frame, cropped_frame, cv::COLOR_BGR2RGB);
        cropped_frame.convertTo(cropped_frame, CV_32F, 1.0 / 255);
        auto input_tensor = torch::from_blob(cropped_frame.data, {1, cropped_frame.rows, cropped_frame.cols, 3}).permute({0, 3, 1, 2}).to(device);

        // Perform inference
        auto output = model.forward({input_tensor}).toTuple();

        // Parse results (simplified)
        // Note: The parsing depends on your model's output format
        // You may need to adjust this part to match your model

        // For demonstration purposes, we'll assume the output provides bounding boxes
        // You will need to implement the parsing logic based on your model's output

        // Move the mouse cursor to the center of the detected player
        // int screen_center_x = left + center_x;
        // int screen_center_y = top + center_y;
        // kmbox.move(screen_center_x, screen_center_y);

        // Control the loop rate
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    cap.release();
}

void hotkey_listener(std::atomic<bool>& run_event, std::atomic<bool>& detector_running) {
    // Define the hotkey (platform-specific)
#ifdef _WIN32
    cout << "Press 'Ctrl + Alt + T' to toggle the detector on/off." << endl;
    while (true) {
        if ((GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_MENU) & 0x8000) && (GetAsyncKeyState('T') & 0x8000)) {
            if (run_event.load()) {
                run_event.store(false);
                detector_running.store(false);
                cout << "Detector paused." << endl;
            } else {
                run_event.store(true);
                detector_running.store(true);
                cout << "Detector running." << endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Debounce
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
#else
    // Implement for Linux or macOS
#endif
}

void sdl_thread(std::atomic<bool>& run_event, std::atomic<bool>& detector_running) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        cerr << "[ERROR] SDL_Init Error: " << SDL_GetError() << endl;
        return;
    }

    SDL_Window* window = SDL_CreateWindow("FNAB", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 300, 200, SDL_WINDOW_SHOWN);
    if (window == nullptr) {
        cerr << "[ERROR] SDL_CreateWindow Error: " << SDL_GetError() << endl;
        SDL_Quit();
        return;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (renderer == nullptr) {
        cerr << "[ERROR] SDL_CreateRenderer Error: " << SDL_GetError() << endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

    bool running = true;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                run_event.store(false);
                running = false;
            } else if (e.type == SDL_MOUSEBUTTONDOWN) {
                int x, y;
                SDL_GetMouseState(&x, &y);
                // Check if mouse is over the button (implement button logic)
                // If so, toggle detector_running and run_event
                detector_running.store(!detector_running.load());
                run_event.store(detector_running.load());
            }
        }

        // Render GUI (implement GUI rendering logic)

        SDL_RenderClear(renderer);
        // Draw GUI elements here
        SDL_RenderPresent(renderer);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

int main() {
    // Set up the run event for the detector thread
    std::atomic<bool> run_event(true);
    std::atomic<bool> detector_running(true);

    // Initialize KMBOX
    Kmbox kmbox;
    // Check if KMBOX is connected
    // Implement a method in Kmbox class to check connection status

    // Define the detection area (250x250 square in the middle of the frame)
    int area_width = 250;
    int area_height = 250;

    // Assuming the capture card provides frames of size 1920x1080
    int frame_width = 1920; // Change this to match your capture card's output
    int frame_height = 1080; // Change this to match your capture card's output

    int left = (frame_width - area_width) / 2;
    int top = (frame_height - area_height) / 2;

    cv::Rect detection_area(left, top, area_width, area_height);

    // Start the detector thread
    std::thread detector(detector_thread, std::ref(run_event), std::ref(detector_running), std::ref(kmbox), detection_area);

    // Start the hotkey listener thread
    std::thread hotkey(hotkey_listener, std::ref(run_event), std::ref(detector_running));

    // Start the SDL (GUI) thread
    sdl_thread(run_event, detector_running);

    // Wait for threads to finish
    detector.join();
    hotkey.join();

    return 0;
}
