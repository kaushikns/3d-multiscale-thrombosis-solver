#!/bin/bash

TARGET=cylinder.stl
INLET_SOURCE=inlet.stl
OUTLET_SOURCE=outlet.stl
WALL_SOURCE=wall.stl

[ -f $TARGET	 ] && rm $TARGET
touch $TARGET

sed -i '1 s/^.*$/solid inlet/' $INLET_SOURCE
cat $INLET_SOURCE >> $TARGET

sed -i '1 s/^.*$/solid outlet/' $OUTLET_SOURCE
cat $OUTLET_SOURCE >> $TARGET

sed -i '1 s/^.*$/solid wall/' $WALL_SOURCE
cat $WALL_SOURCE >> $TARGET
