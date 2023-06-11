#!/bin/bash

set -e

while getopts ":o:d:" opt; do
  case $opt in
    o) DATA_DIR="$OPTARG"
    ;;
    d) OUTPUT_DIR="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    exit 1
    ;;
  esac

  case $OPTARG in
    -*) echo "Option $opt needs a valid argument"
    exit 1
    ;;
  esac
done

if [ -z "$DATA_DIR" ]
then
    echo "-d can't be empty, need to provide data directory"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]
then
    echo "-o can't be empty, need to provide output directory"
    exit 1
fi
