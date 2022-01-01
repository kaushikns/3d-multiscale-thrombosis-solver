#include <iostream>

template <typename T, std::size_t N_SYMBOLS>
inline T grayEncode(const T b)
{
	T res = 0;
	for (std::size_t i = 0; i < N_SYMBOLS; ++i)
	{
		const T blkSize = 1 << i;
		const T value =
			( (b >= blkSize) &&
			  ( (((b-blkSize)/(2*blkSize)) % 2) == 0u) ) ? 1 : 0;
		res |= value << i;
	}

	return res;
}


template <typename T, std::size_t N_SYMBOLS>
T grayDecode(const T n)
{
	bool invert = false;

	T res = 0;
	for (int i = N_SYMBOLS-1; i >= 0; --i)
	{
		if (invert)
			res |= (~n & (1 << i));
		else
			res |= (n & (1 << i));
		invert = res & (1 << i); // bit set ?
	}

	return res;
}

int main()
{
	for (int i = 0; i < 1<<4; ++i)
	{
		std::cout << i << " " << grayDecode<int, 4>(grayEncode<int, 4>(i))
			<< " " << grayEncode<int, 4>(i) << std::endl;
	}

	return 0;
}
