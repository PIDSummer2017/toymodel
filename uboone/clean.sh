#!/usr/bin/env bash

args=("$@")

#echo Number of arguments: $#
echo Removing csv and weight of: ${args[0]}

rm test_csv/plane0/${args[0]}/*csv
rm test_csv/plane1/${args[0]}/*csv
rm test_csv/plane2/${args[0]}/*csv

rm plane2training/${args[0]}/${args[0]}*
rm plane2training/${args[0]}/${args[0]}*

