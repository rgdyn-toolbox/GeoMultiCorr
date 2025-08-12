#!/bin/bash
# Creating directory for ASP installation
mkdir $HOME/ASP_install
# Changing to the ASP installation directory
cd $HOME/ASP_install
# Downloading the ASP installation script
if [ ! -f StereoPipeline-3.5.0-2025-04-28-x86_64-Linux.tar.bz2 ]; then
	wget https://github.com/NeoGeographyToolkit/StereoPipeline/releases/download/v3.5.0/StereoPipeline-3.5.0-2025-04-28-x86_64-Linux.tar.bz2
fi
tar -xvf StereoPipeline-3.5.0-2025-04-28-x86_64-Linux.tar.bz2
# Adding ASP binaries to PATH
export PATH="${PATH}":"$HOME/ASP_install/StereoPipeline-3.5.0-2025-04-28-x86_64-Linux/bin"
# Chekking if ASP is installed correctly
if command -v stereo >/dev/null 2>&1; then
    echo "ASP installed successfully."
else
    echo "ASP installation failed."
    exit 1
fi
