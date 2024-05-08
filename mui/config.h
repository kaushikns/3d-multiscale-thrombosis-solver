/*****************************************************************************
* Multiscale Universal Interface Code Coupling Library                       *
*                                                                            *
* Copyright (C) 2019 Y. H. Tang, S. Kudo, X. Bian, Z. Li, G. E. Karniadakis  *
*                                                                            *
* This software is jointly licensed under the Apache License, Version 2.0    *
* and the GNU General Public License version 3, you may use it according     *
* to either.                                                                 *
*                                                                            *
* ** Apache License, version 2.0 **                                          *
*                                                                            *
* Licensed under the Apache License, Version 2.0 (the "License");            *
* you may not use this file except in compliance with the License.           *
* You may obtain a copy of the License at                                    *
*                                                                            *
* http://www.apache.org/licenses/LICENSE-2.0                                 *
*                                                                            *
* Unless required by applicable law or agreed to in writing, software        *
* distributed under the License is distributed on an "AS IS" BASIS,          *
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
* See the License for the specific language governing permissions and        *
* limitations under the License.                                             *
*                                                                            *
* ** GNU General Public License, version 3 **                                *
*                                                                            *
* This program is free software: you can redistribute it and/or modify       *
* it under the terms of the GNU General Public License as published by       *
* the Free Software Foundation, either version 3 of the License, or          *
* (at your option) any later version.                                        *
*                                                                            *
* This program is distributed in the hope that it will be useful,            *
* but WITHOUT ANY WARRANTY; without even the implied warranty of             *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
* GNU General Public License for more details.                               *
*                                                                            *
* You should have received a copy of the GNU General Public License          *
* along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
*****************************************************************************/

/**
 * @file config.h
 * @author S. Kudo
 * @date 10 March 2013
 * @brief File containing data structures defining all data types used
 * by an interface.
 */

#ifndef CONFIG_H
#define CONFIG_H

#include "util.h"
#include "dim.h"
#include "exception.h"

namespace mui
{

template <typename... TYPES>
struct type_list
{
};

struct sendplatelet
{
	double px, py, pz, release, calcInteg;
	long long loc;
	int rank;
};

struct recvplatelet
{
	long long loc;
	double adpC, txa2C;
	int rank;
};

struct sendplateletlb
{
	int cx, cy, cz;
};

mui::ostream &operator<<(mui::ostream &ost, const sendplatelet &plt)
{
	return ost << plt.px << plt.py << plt.pz << plt.release << plt.calcInteg << plt.loc << plt.rank;
}

mui::istream &operator>>(mui::istream &ist, sendplatelet &plt)
{
	return ist >> plt.px >> plt.py >> plt.pz >> plt.release >> plt.calcInteg >> plt.loc >> plt.rank;
}

mui::ostream &operator<<(mui::ostream &ost, const recvplatelet &plt)
{
	return ost << plt.loc << plt.adpC << plt.txa2C << plt.rank;
}

mui::istream &operator>>(mui::istream &ist, recvplatelet &plt)
{
	return ist >> plt.loc >> plt.adpC >> plt.txa2C >> plt.rank;
}

mui::ostream &operator<<(mui::ostream &ost, const sendplateletlb &plt)
{
	return ost << plt.cx << plt.cy << plt.cz;
}

mui::istream &operator>>(mui::istream &ist, sendplateletlb &plt)
{
	return ist >> plt.cx >> plt.cy >> plt.cz;
}

struct one_dim
{
	using EXCEPTION = exception_segv; //- Exception handling type

	static const bool DEBUG = false;	   //- Enable extra debugging output
	static const int D = 1;				   //- Dimensionality of the domain
	static const bool FIXEDPOINTS = false; //- Optimise for fixed structures - points, and order in which parameters sent must be unchanging between time frames
	static const bool QUIET = false;	   //- If the library is quiet then it will only issue critical warning messages

	using REAL = double; //- REAL data type
	using INT = int;	 //- INT data type

	using time_type = REAL;				  //- time_type INT for iteration coupling, REAL for exact coupling
	using point_type = point<REAL, D>;	  //- "point" data type and dimensionality
	using data_types = type_list<int32_t, //- Data types that can be used in the interface
								 int64_t,
								 double,
								 float,
								 std::string,
								 struct sendplatelet,
								 struct recvplatelet,
								 struct sendplateletlb>;
};

struct two_dim
{
	using EXCEPTION = exception_segv;

	static const bool DEBUG = false;
	static const int D = 2;
	static const bool FIXEDPOINTS = false;
	static const bool QUIET = false;

	using REAL = double;
	using INT = int;

	using point_type = point<REAL, D>;
	using time_type = REAL;
	using data_types = type_list<int32_t,
								 int64_t,
								 double,
								 float,
								 std::string>;
};

struct three_dim
{
	using EXCEPTION = exception_segv;

	static const bool DEBUG = false;
	static const int D = 3;
	static const bool FIXEDPOINTS = false;
	static const bool QUIET = false;

	using REAL = double;
	using INT = int;

	using point_type = point<REAL, D>;
	using time_type = REAL;
	using data_types = type_list<int32_t,
								 int64_t,
								 double,
								 float,
								 std::string>;
};

// backward-compatibility
struct default_config : three_dim
{
};
} // namespace mui

#endif
