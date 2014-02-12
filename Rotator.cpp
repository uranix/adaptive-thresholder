#include <Magick++.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <vector>
#include <cmath>

using namespace std;
using namespace Magick;

int usage(string argv0) {
	cerr << "USAGE: " << argv0 << " <input_image> <output_image> [maxangle] [intervals]" << endl;
	return 1;
}

class ImageWrapper {
	const PixelPacket *pp;
	const int colorShift;
	const int width, height;
public:
	ImageWrapper(const Image &image) 
	:
		colorShift(8 * (sizeof(pp->green) - sizeof(uint8_t))),
		width(image.size().width()), 
		height(image.size().height())
	{
		pp = image.getConstPixels(0, 0, width, height);
	}
	uint8_t gray(int x, int y) const {
		int v = (*this)(x, y)->green >> colorShift;
		assert(v < 256);
		assert(v >= 0);
		return v;
	}
	const PixelPacket *operator()(int x, int y) const {
		if (x < 0)
			x += width;
		if (y < 0)
			y += height;
		if (x >= width)
			x -= width;
		if (y >= height)
			y -= height;
		assert(x >= 0 && y >= 0 && x < width && y < height);
		
		return pp + y * width + x;
	}
	int isblack(int x, int y) const {
		return gray(x, y) < 128;
	}
	int getColorShift() const {
		return colorShift;
	}
};

int lineScore(const Image &input, int deltay) {
	Geometry g = input.size();
	const int width = g.width();
	const int height = g.height();

	ImageWrapper iw(input);

	std::vector<int> lineTotal(height, 0);

	int dir = deltay > 0 ? 1 : (deltay < 0 ? -1 : 0);
	int deltaerr = deltay * dir;

	int cnt = 0;

	for (int y0 = 0; y0 < height; y0++) {
		int y = y0;
		int error = 0;
		lineTotal[y0] = 0;
		for (int x = 0; x < width; x++) {
			lineTotal[y0] += iw.isblack(x, y);
			error += deltaerr;
			if (2 * error >= width) {
				y += dir;
				error -= width;
			}
		}
		if (lineTotal[y0])
			cnt++;
	}
	return cnt;
}

void tilt(const Image &input, Image &output, int deltay) {
	Geometry g = input.size();
	const int width = g.width();
	const int height = g.height();

	ImageWrapper iw(input);

	int dir = deltay > 0 ? 1 : (deltay < 0 ? -1 : 0);
	int deltaerr = deltay * dir;

	for (int y0 = 0; y0 < height; y0++) {
		int y = y0;
		int error = 0;
		for (int x = 0; x < width; x++) {
			*output.getPixels(x, y0, 1, 1) = *iw(x, y);
			error += deltaerr;
			if (2 * error >= width) {
				y += dir;
				error -= width;
			}
		}
	}
}

int main(int argc, char **argv)
{
	if (argc < 3 || argc > 5)
		return usage(argv[0]);

	int cols = 21;
	double maxangle = 10;

	if (argc > 3)
		maxangle = atof(argv[3]);
	
	if (argc > 4)
		cols = atoi(argv[4]);

	cout << "Processing `" << argv[1] << "' into `" << argv[2] << "'" << " with angular range from -" << maxangle << " to " << maxangle
		<< " degrees and " << cols << " intervals" << endl;

	InitializeMagick(*argv);
	Image input;
	Color black("black");

	try {
		input.read(argv[1]);
		input.quantizeColorSpace(GRAYColorspace);
		input.quantizeColors(256);
		input.quantize();


		std::vector<int> dy(cols);
		std::vector<int> score(cols);

		Image output(input.size(), "white");

		int width = input.size().width();

		int minpos = 0;
		double range = tan(maxangle / 45 * atan(1.));

		for (int i = 0; i < cols; i++) {
			dy[i] = range * width * (2 * i - cols + 1) / (cols - 1);
			score[i] = lineScore(input, dy[i]);

			if (score[i] < score[minpos])
				minpos = i;
		}

		if (minpos == 0 || minpos == (cols - 1)) {
			cerr << "Angle range was too narrow" << endl;
			return 2;
		}

		int a = dy[minpos - 1];
		int b = dy[minpos + 1];

		double w = 0.381966;

		int c = a + w * (b - a);
		int d = a + b - c;

		int fc = lineScore(input, c);
		int fd = lineScore(input, d);
		
		while (b > d && d > c && c > a) {
			if (fd > fc) {
				b = d;
				d = c;
				c = a + b - d;
				fd = fc;
				fc = lineScore(input, c);
			} else {
				a = c;
				c = d;
				d = a + b - c;
				fc = fd;
				fd = lineScore(input, d);
			}
		}

		int fa;
		if (fd < fc) {
			minpos = d;
			fa = fd;
		} else {
			minpos = c;
			fa = fc;
		}

		for (int i = std::min(d, c); i <= std::max(d, c) + 1; i++) {
			int v = lineScore(input, i);
			if (v < fa) {
				fa = v;
				minpos = i;
			}
		}

		cout << "Optimum tilt = " << minpos << "px" << endl;

		tilt(input, output, minpos);
		output.write(argv[2]);
	} catch (Exception &e) { 
		cerr << "Caught exception: " << e.what() << endl;
		return 1;
	}
	return 0;
}
