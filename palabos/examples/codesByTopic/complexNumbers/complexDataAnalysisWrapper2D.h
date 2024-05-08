/* This file is part of the Palabos library.
 *
 * The Palabos softare is developed since 2011 by FlowKit-Numeca Group Sarl
 * (Switzerland) and the University of Geneva (Switzerland), which jointly
 * own the IP rights for most of the code base. Since October 2019, the
 * Palabos project is maintained by the University of Geneva and accepts
 * source code contributions from the community.
 * 
 * Contact:
 * Jonas Latt
 * Computer Science Department
 * University of Geneva
 * 7 Route de Drize
 * 1227 Carouge, Switzerland
 * jonas.latt@unige.ch
 *
 * The most recent release of Palabos can be downloaded at 
 * <https://palabos.unige.ch/>
 *
 * The library Palabos is free software: you can redistribute it and/or
 * modify it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * The library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/** \file
 * Helper functions for domain initialization -- header file.
 */
#ifndef COMPLEX_DATA_ANALYSIS_WRAPPER_2D_H
#define COMPLEX_DATA_ANALYSIS_WRAPPER_2D_H

#include "core/globalDefs.h"
#include "atomicBlock/blockLattice2D.h"
#include "atomicBlock/dataField2D.h"
#include "multiBlock/multiBlockLattice2D.h"
#include "multiBlock/multiDataField2D.h"
#include "typeConverterFunctional2D.h"
#include <memory>


namespace plb {

/* ******************************************************************* */
/* *************** PART II. Atomic-block wrappers: Scalar-Field ****** */
/* ******************************************************************* */

template<typename T, typename U>
void realPart( ScalarField2D<T>& complexField,
               ScalarField2D<U>& realField,
               Box2D domain);

template<typename T, typename U>
std::unique_ptr<ScalarField2D<U> > realPart(ScalarField2D<T>& complexField, Box2D domain);
    
template<typename T, typename U>
void imaginaryPart( ScalarField2D<T>& complexField,
                    ScalarField2D<U>& realField,
                    Box2D domain);

template<typename T, typename U>
std::unique_ptr<ScalarField2D<U> > imaginaryPart(ScalarField2D<T>& complexField, Box2D domain);

/* ******************************************************************* */
/* *************** PART III. Atomic-block wrappers: Tensor-field ***** */
/* ******************************************************************* */

template<typename T, typename R, plint d>
void realPart( TensorField2D<T,d>& complexField,
               TensorField2D<R,d>& realField,
               Box2D domain);

template<typename T, typename R, int d>
std::unique_ptr<TensorField2D<R,d> > realPart(TensorField2D<T,d>& complexField, Box2D domain);

template<typename T, typename R, int d>
void imaginaryPart( TensorField2D<T,d>& complexField,
                    TensorField2D<R,d>& realField,
                    Box2D domain);

template<typename T, typename R, int d>
std::unique_ptr<TensorField2D<R,d> > imaginaryPart(TensorField2D<T,d>& complexField, Box2D domain); 


/* *********************************************************************** */
/* *************** PART V : Multi-block wrappers: Multi-Scalar-Field ***** */
/* *********************************************************************** */
	
/* *************** From Complex to real fields ******************************* */

template<typename T, typename U>
void realPart( MultiScalarField2D<T>& complexField,
               MultiScalarField2D<U>& realField,
               Box2D domain);

template<typename T, typename U>
std::unique_ptr<MultiScalarField2D<U> > realPart(MultiScalarField2D<T>& complexField, Box2D domain);
	
template<typename T, typename U>
void imaginaryPart( MultiScalarField2D<T>& complexField,
                    MultiScalarField2D<U>& realField,
                    Box2D domain);

template<typename T, typename U>
std::unique_ptr<MultiScalarField2D<U> > imaginaryPart(MultiScalarField2D<T>& complexField, Box2D domain);

/* *********************************************************************** */
/* ************** PART VI : Multi-block wrappers: Multi-Tensor-Field ***** */
/* *********************************************************************** */

template<typename T, typename R, plint d>
void realPart( MultiTensorField2D<T,d>& complexField,
               MultiTensorField2D<R,d>& realField,
               Box2D domain);

template<typename T, typename R, int d>
std::unique_ptr<MultiTensorField2D<R,d> > realPart(MultiTensorField2D<T,d>& complexField, Box2D domain);

template<typename T, typename R, int d>
void imaginaryPart( MultiTensorField2D<T,d>& complexField,
                    MultiTensorField2D<R,d>& realField,
                    Box2D domain);

template<typename T, typename R, int d>
std::unique_ptr<MultiTensorField2D<R,d> > imaginaryPart(MultiTensorField2D<T,d>& complexField, Box2D domain);	
	
}  // namespace plb

#endif  // COMPLEX_DATA_ANALYSIS_WRAPPER_2D_H
