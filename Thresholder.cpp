#include <Magick++.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include <cassert>
#include <cstring>

#ifdef USE_OMP
# include <omp.h>
#else
int omp_get_thread_num() {
	return 0;
}
#endif

using namespace std;
using namespace Magick;

int usage(string argv0) {
	cerr << "USAGE: " << argv0 << " <input_image> <output_image> [window size] [level]" << endl;
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
	uint8_t operator()(int x, int y) const {
		if (x < 0)
			x = -x;
		if (y < 0)
			y = -y;
		if (x >= width)
			x = 2 * width - 1 - x;
		if (y >= height)
			y = 2 * height - 1 - y;
		assert(x >= 0 && y >= 0 && x < width && y < height);
		const PixelPacket *pixel = pp + y * width + x;
		int v = pixel->green >> colorShift;
		assert(v < 256);
		assert(v >= 0);
		return v;
	}
	int getColorShift() const {
		return colorShift;
	}
};

class Histogramm {
	uint32_t data[256];
	const int blockSizeX, blockSizeY;
	const ImageWrapper &iw;

	double black_mean;
	double white_mean;

	int x0, y0;

	double lev;

	void init() {
		for (int i = 0; i < 256; i++)
			data[i] = 0;
		for (int x = x0 - blockSizeX; x < x0 + blockSizeX; x++)
			for (int y = y0 - blockSizeY; y < y0 + blockSizeY; y++)
				data[iw(x, y)]++;
	}
	void remove_left() {
		int x = x0 - blockSizeX;
		for (int y = y0 - blockSizeY; y < y0 + blockSizeY; y++)
			data[iw(x, y)]--;
	}
	void add_right() {
		int x = x0 + blockSizeX;
		for (int y = y0 - blockSizeY; y < y0 + blockSizeY; y++)
			data[iw(x, y)]++;
	}
public:
	Histogramm(const ImageWrapper &input, const int blockSizeX, const int blockSizeY, int x0, int y0, double lev)
		: blockSizeX(blockSizeX), blockSizeY(blockSizeY),
		iw(input), x0(x0), y0(y0), lev(lev)
	{
		black_mean = 0;
		white_mean = 255;
		init();
	}
	void advance() {
		remove_left();
		add_right();
		x0++;
	}
	double fit() {
		int iprev = -1;
		bool done = false;
		int it, itmax = 20;
		
		int black_cnt, white_cnt;

		it = itmax;
		black_mean = 0;

		int edge;

		while (!done) {
			edge = (black_mean + white_mean) / 2;
			if (edge < 0)
				edge = 0;

			if (edge == iprev)
				done = true;
			
			iprev = edge;
			it --;
			if (!it)
				done = true;

			black_mean = white_mean = 0;
			black_cnt = white_cnt = 0;

			for (int i = 0; i < edge; i++) {
				black_mean += i * data[i];
				black_cnt += data[i];
			}

			for (int i = edge; i < 256; i++) {
				white_mean += i * data[i];
				white_cnt += data[i];
			}

			if (black_cnt)
				black_mean /= black_cnt;
			else {
				black_mean = -255;
				done = true;
			}
			
			if (white_cnt)
				white_mean /= white_cnt;
			else
				white_mean = edge - black_mean / 2;
		}
		double ret = (1 - lev) * black_mean + lev * white_mean;
		if (ret < 0)
			return 0;
		return ret;
	}
};

void process(const Image &input, Image &output, const int blockSizeX, const int blockSizeY, double lev) {
	Geometry g = input.size();
	const int width = g.width();
	const int height = g.height();

	Color black("black");

	ImageWrapper iw(input);

	int progress = 0;
	int mark = 0;
	int tics = 75;
	int step = height / tics;

	for (int i = 0; i < tics; i++)
		cout << ' ';
	cout << "  ]\r[";

	#pragma omp parallel for schedule(dynamic)
	for (int y = 0; y < height; y++) {
		Histogramm h(iw, blockSizeX, blockSizeY, 0, y, lev);
		if (omp_get_thread_num() == 0) {
			while (progress > mark) {
				cout << '=';
				cout.flush();
				mark += step;
			}
		}
		for (int x = 0; x < width; x++) {
			double level = h.fit();

			if (iw(x, y) < level)
				*output.getPixels(x, y, 1, 1) = black;

			h.advance();
		}
		#pragma omp atomic
		progress++;
	}
	cout << "]" << endl;
}

int main(int argc, char **argv)
{
	int blockSizeX = 32;
	int blockSizeY = 2;
	double lev = .5;

	if (argc < 3 || argc > 5)
		return usage(argv[0]);
	
	if (argc > 3) {
		char *p;
		if ((p = strchr(argv[3], 'x'))) {
			blockSizeX = atoi(argv[3]);
			blockSizeY = atoi(p + 1);
		} else
			blockSizeX = blockSizeY = atoi(argv[3]);
	}
	
	if (argc > 4) {
		if (strchr(argv[4], '%'))
			lev = 0.01 * atof(argv[4]);
		else
			lev = atof(argv[4]);
	}

	cout << "Processing `" << argv[1] << "' into `" << argv[2] 
		<< "' with window " << blockSizeX << "x" << blockSizeY
		<< " and threshold " << lev * 100 << "%" << endl;

	InitializeMagick(*argv);
	Image input;

	try {
		input.read(argv[1]);
		input.quantizeColorSpace(GRAYColorspace);
		input.quantizeColors(256);
		input.quantize();

		Image output(input.size(), "white");

		process(input, output, blockSizeX, blockSizeY, lev);

		output.write(argv[2]);
	} catch (Exception &e) { 
		cerr << "Caught exception: " << e.what() << endl;
		return 1;
	}
	return 0;
}
