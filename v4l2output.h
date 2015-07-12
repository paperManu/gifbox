#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <iostream>
#include <string>

#include <linux/videodev2.h>

/*************/
class V4l2Output
{
    public:
        V4l2Output(int width, int height, std::string device = "/dev/video0");
        ~V4l2Output();

        explicit operator bool() const
        {
            if (_sink < 0)
                return false;
            else
                return true;
        }

        bool writeToDevice(void* data, size_t size);

    private:
        std::string _device {};
        int _sink {-1};
};
