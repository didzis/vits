#!/bin/sh
#
# Script for downloading Meta MMS TTS models: https://github.com/facebookresearch/fairseq/tree/main/examples/mms
#
# List of language codes: https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html
#

scriptdir="$(cd "`dirname "$0"`"; pwd)"

if [ -z "$1" ]; then
    echo "Script for downloading Meta MMS TTS models"
    echo
    echo "List of language codes: https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html"
    echo
    echo "usage: $0 LANG [TARGET-DIR]"
    exit 1
fi

lang="$1"
targetdir="$2"
[ -z "$2" ] && targetdir="$scriptdir"

echo "Downloading Meta MMS TTS model for language $lang"
curl -O https://dl.fbaipublicfiles.com/mms/tts/${lang}.tar.gz -f
if [ $? -ne 0 ]; then
    rm -f ${lang}.tar.gz
    echo "failed to download"
    exit 1
fi

echo "Extracting to $targetdir"
mkdir -p "$targetdir"
tar xvzf ${lang}.tar.gz -C "$targetdir"
if [ $? -eq 0 ]; then
    rm -f ${lang}.tar.gz
    echo "ok"
else
    echo "failed to extract"
    exit 1
fi
