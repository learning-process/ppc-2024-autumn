#!/bin/bash

echo "set debuginfod enabled off" > ~/.gdbinit

printf "xterm*faceName: DejaVuSansMon\nxterm*faceSize: 11\n" > ~/.Xresources
echo " [[ -e ~/.Xresources ]] && xrdb ~/.Xresources" >> ~/.bashrc