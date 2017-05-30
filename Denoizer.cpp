#include <Magick++.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include <cassert>
#include <cstring>
#include <vector>
#include <cmath>
#include <fftw3.h>

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

void mult(fftw_complex &a, const fftw_complex &b) {
	double ar = a[0];
	double ai = a[1];
	double br = b[0];
	double bi = b[1];

	a[0] = ar * br - ai * bi;
	a[1] = ai * br + ar * bi;
}

void mult(fftw_complex *a, const fftw_complex *b, int hwidth, int height) {
	for (int i = 0; i < hwidth * height; i++)
		mult(a[i], b[i]);
}

int main(int argc, char **argv)
{
	if (argc < 3 || argc > 5)
		return usage(argv[0]);

	cout << "Processing `" << argv[1] << "' into `" << argv[2] << "'" << endl;

	InitializeMagick(*argv);
	Image input;

	Color black("black");
	Color white("white");

	double Rmax = 30;
	double Rmin = 10;

	try {
		input.read(argv[1]);
		input.quantizeColorSpace(GRAYColorspace);
		input.quantizeColors(256);
		input.quantize();
		Geometry g = input.size();
		const int width = g.width();
		const int height = g.height();

		ImageWrapper iw(input);

		std::vector<double> in(width * height);
		std::vector<double> k1(width * height);
		std::vector<double> k2(width * height);
		int k1cnt = 0, k2cnt = 0;

		fftw_init_threads();
		fftw_plan_with_nthreads(4);

		fftw_complex *fft = static_cast<fftw_complex *>(fftw_malloc((width / 2 + 1) * height * sizeof(fftw_complex)));
		fftw_complex *k1_fft = static_cast<fftw_complex *>(fftw_malloc((width / 2 + 1) * height * sizeof(fftw_complex)));
		fftw_complex *k2_fft = static_cast<fftw_complex *>(fftw_malloc((width / 2 + 1) * height * sizeof(fftw_complex)));
		int z = 0;
		for (int y = 0; y < height; y++) 
			for (int x = 0; x < width; x++) {
				in[z] = 1. / 255 * iw.gray(x, y);
				
				int dx = std::min(x, width - x);
				int dy = std::min(y, height - y);
				int dr2 = dx * dx + dy * dy;

				if (dr2 > Rmin * Rmin && dr2 < Rmax * Rmax) {
					k1[z] = 1;
					k1cnt++;
				} else
					k1[z] = 0;
				
				if (dr2 <= Rmin * Rmin) {
					k2[z] = 1;
					k2cnt++;
				} else
					k2[z] = 0;
				
				z++;
		}

		fftw_plan planForw = fftw_plan_dft_r2c_2d(height, width, &in[0], &fft[0], FFTW_ESTIMATE);
		fftw_execute(planForw);
		fftw_plan planForw1 = fftw_plan_dft_r2c_2d(height, width, &k1[0], &k1_fft[0], FFTW_ESTIMATE);
		fftw_execute(planForw1);
		fftw_plan planForw2 = fftw_plan_dft_r2c_2d(height, width, &k2[0], &k2_fft[0], FFTW_ESTIMATE);
		fftw_execute(planForw2);

		mult(k1_fft, fft, width / 2 + 1, height);
		mult(k2_fft, fft, width / 2 + 1, height);

		fftw_plan planBack1 = fftw_plan_dft_c2r_2d(height, width, &k1_fft[0], &k1[0], FFTW_ESTIMATE);
		fftw_execute(planBack1);
		fftw_plan planBack2 = fftw_plan_dft_c2r_2d(height, width, &k2_fft[0], &k2[0], FFTW_ESTIMATE);
		fftw_execute(planBack2);

		fftw_destroy_plan(planForw);
		fftw_destroy_plan(planForw1);
		fftw_destroy_plan(planForw2);
		fftw_destroy_plan(planBack1);
		fftw_destroy_plan(planBack2);

		fftw_free(fft);
		fftw_free(k1_fft);
		fftw_free(k2_fft);

		Image output(input.size(), "white");
		
		z = 0;

		int N = width * height;

#ifdef DEBUG
		Image core(input.size(), "white");
		Image ring(input.size(), "white");
		PixelPacket *cpp = core.setPixels(0, 0, width, height);
		PixelPacket *rpp = ring.setPixels(0, 0, width, height);
#endif

		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				k1[z] /= 1. * N * k1cnt;
				k2[z] /= 1. * N * k2cnt;

#ifdef DEBUG
				rpp[z] = ColorGray(k1[z]);
				cpp[z] = ColorGray(k2[z]);
#endif

				z++;
			}
#ifdef DEBUG
		core.syncPixels();
		ring.syncPixels();
		core.write("core.png");
		ring.write("ring.png");
#endif

		z = 0;
		PixelPacket *opp = output.setPixels(0, 0, width, height);
		const PixelPacket *ipp = input.getConstPixels(0, 0, width, height);
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				opp[z] = ipp[z];
				z++;
			}
		output.syncPixels();

		output.strokeWidth(0);
		output.fillColor("white");
		int rmin = Rmin + 0.5;
		z = 0;
		for (int y = 0; y < height; y++)
			for (int x = 0; x < width; x++) {
				if (k1[z] > 0.999 && k2[z] < 0.999)
					output.draw(DrawableCircle(x, y, x - rmin, y));
				z++;
			}

		output.write(argv[2]);
	} catch (Exception &e) { 
		cerr << "Caught exception: " << e.what() << endl;
		return 1;
	}
	return 0;
}
