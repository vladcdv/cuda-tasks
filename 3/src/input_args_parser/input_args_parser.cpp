#include "input_args_parser.h"
#include <iostream>
#include "../utils/version.h"

namespace cuda_filter
{

    InputArgsParser::InputArgsParser(int argc, char **argv)
        : m_argc(argc), m_argv(argv)
    {
    }

    InputSource InputArgsParser::stringToInputSource(const std::string &str)
    {
        if (str == "webcam")
            return InputSource::WEBCAM;
        if (str == "image")
            return InputSource::IMAGE;
        if (str == "video")
            return InputSource::VIDEO;
        if (str == "synthetic")
            return InputSource::SYNTHETIC;
        throw std::runtime_error("Invalid input source: " + str);
    }

    SyntheticPattern InputArgsParser::stringToSyntheticPattern(const std::string &str)
    {
        if (str == "checkerboard")
            return SyntheticPattern::CHECKERBOARD;
        if (str == "gradient")
            return SyntheticPattern::GRADIENT;
        if (str == "noise")
            return SyntheticPattern::NOISE;
        throw std::runtime_error("Invalid synthetic pattern: " + str);
    }

    ToneMapper InputArgsParser::stringToToneMapper(const std::string &str)
    {
        if (str == "none")
            return ToneMapper::NONE;
        if (str == "reinhard")
            return ToneMapper::REINHARD;
        if (str == "filmic")
            return ToneMapper::FILMIC;
        throw std::runtime_error("Invalid tonemapper: " + str);
    }

    FilterOptions InputArgsParser::parseArgs()
    {
        cxxopts::Options options("cuda-webcam-filter", "Real-time webcam filter with CUDA acceleration");

        setupOptions(options);

        auto result = options.parse(m_argc, m_argv);

        if (result.count("help"))
        {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        if (result.count("version"))
        {
            std::cout << "CUDA Webcam Filter version " << CUDA_WEBCAM_FILTER_VERSION << std::endl;
            exit(0);
        }

        FilterOptions filterOptions;

        // Parse input source
        std::string inputType = result["input"].as<std::string>();
        filterOptions.inputSource = stringToInputSource(inputType);
        filterOptions.inputPath = result["path"].as<std::string>();

        if (filterOptions.inputSource == InputSource::SYNTHETIC)
        {
            std::string patternType = result["synthetic"].as<std::string>();
            filterOptions.syntheticPattern = stringToSyntheticPattern(patternType);
        }
        else if (filterOptions.inputSource == InputSource::WEBCAM)
        {
            filterOptions.deviceId = result["device"].as<int>();
        }

        filterOptions.filterType = result["filter"].as<std::string>();
        filterOptions.kernelSize = result["kernel-size"].as<int>();
        filterOptions.sigma = result["sigma"].as<float>();
        filterOptions.intensity = result["intensity"].as<float>();
        filterOptions.preview = result.count("preview") > 0;

        // Parse HDR CLI arguments
        filterOptions.exposure = result["exposure"].as<float>();
        filterOptions.gamma = result["gamma"].as<float>();
        filterOptions.saturation = result["saturation"].as<float>();
        filterOptions.toneMapper = stringToToneMapper(result["tonemap"].as<std::string>());

        return filterOptions;
    }

    void InputArgsParser::setupOptions(cxxopts::Options &options)
    {
        options.add_options()("i,input", "Input source: 'webcam', 'image', 'video', or 'synthetic'",
                              cxxopts::value<std::string>()->default_value("webcam"))("p,path", "Path to input image or video file (when not using webcam)",
                                                                                      cxxopts::value<std::string>()->default_value("test_image.jpg"))("s,synthetic", "Synthetic pattern type: 'checkerboard', 'gradient', 'noise'",
                                                                                                                                                      cxxopts::value<std::string>()->default_value("checkerboard"))("d,device", "Camera device ID", cxxopts::value<int>()->default_value("0"))("f,filter", "Filter type: blur, sharpen, edge, emboss, hdr", cxxopts::value<std::string>()->default_value("blur"))("k,kernel-size", "Kernel size for filters", cxxopts::value<int>()->default_value("3"))("sigma", "Sigma value for Gaussian blur", cxxopts::value<float>()->default_value("1.0"))("intensity", "Filter intensity", cxxopts::value<float>()->default_value("1.0"))("preview", "Show original video alongside filtered")("h,help", "Print usage")("v,version", "Print version information")("e,exposure", "Exposure multiplier", cxxopts::value<float>()->default_value("1.0f"))("g,gamma", "Gamma", cxxopts::value<float>()->default_value("2.2f"))("saturation", "Saturation", cxxopts::value<float>()->default_value("1.0f"))("t,tonemap", "Tonemapper: 'none', 'reinhard', or 'filmic'", cxxopts::value<std::string>()->default_value("reinhard"));
    }

} // namespace cuda_filter
