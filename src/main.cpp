#include <exception>
#include <iostream>

#include "app.h"
#include "options.h"

int main(int argc, char** argv) {
    using namespace pickdemo;
    try {
        const Options options = parseOptions(argc, argv);
        if (options.help) {
            printUsage(std::cout, argv[0]);
            return 0;
        }
        return run(options);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
