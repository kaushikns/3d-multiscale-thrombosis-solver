/***************************************************************************
 *   Copyright (C) 2007 by BEEKHOF, Fokko                                  *
 *   fpbeekhof@gmail.com                                                   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <cassert>

#include <cvmlcpp/signal/Quantizers>

int main()
{
	using namespace cvmlcpp;

	StaticVector<double, 4> a = 0;

	for (unsigned i = 0; i < (1<<4); ++i)
	{
		graySoftQuantize(double(i) / double((1<<4)-1), a);//, 0.0f);
		std::cout << a.to_string() << std::endl;
	}

	std::cout << std::endl;
// 	for (unsigned i = 0; i < (1<<4); ++i)
// 	{
// 		graySoftQuantize(float(i) / float((1<<4)-1), a, 0.05f);
// 		std::cout << a.to_string() << std::endl;
// 	}

	graySoftQuantize(0.0, a);
	std::cout << a.to_string() << std::endl;
	graySoftQuantize(0.0, a, 0.0005);
	std::cout << a.to_string() << std::endl;

	return 0;
}
