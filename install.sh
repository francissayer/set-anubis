#!/bin/bash

if [[ -r /etc/os-release ]]; then
    . /etc/os-release
    echo "Detected distribution : $NAME ($ID $VERSION_ID)"
else
    echo "Impossible to determine distribution (pas de /etc/os-release)."
    ID=""
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PYTHONPATH="$PYTHONPATH:$script_dir"
echo "Added $script_dir to PYTHONPATH."

if ! grep -q "PYTHONPATH.*$script_dir" ~/.bashrc; then
    echo "PYTHONPATH is not in ~/.bashrc, adding it now..."
    echo "export PYTHONPATH=\$PYTHONPATH:$script_dir" >> ~/.bashrc
    echo "PYTHONPATH updated in ~/.bashrc."
    echo "Sourcing ~/.bashrc..."
    source ~/.bashrc
else
    echo "PYTHONPATH is already set in ~/.bashrc."
fi

if [[ -f "$script_dir/requirements.txt" ]]; then
    echo "Installing Python requirements..."

    PIP_EXTRA_OPTS=()
    if [[ "$ID" == "ubuntu" ]]; then
        echo "Ubuntu detected, uses of --break-system-packages with pip."
        PIP_EXTRA_OPTS+=(--break-system-packages)
    fi

    pip install -r "$script_dir/requirements.txt" "${PIP_EXTRA_OPTS[@]}"
    if [[ $? -ne 0 ]]; then
        echo "Error during installation of Python requirements."
        exit 1
    fi
    echo "Python requirements installed successfully."
else
    echo "No requirements.txt file found. Skipping Python requirements installation."
fi

if [[ $# -gt 0 ]]; then
    software_to_install="$@"
    ./External_Integration/install.sh $software_to_install
    if [[ $? -eq 0 ]]; then
        echo "Installation completed successfully."
    else
        echo "Error during software installation."
        exit 1
    fi
else
    echo "No software specified for installation."
fi
